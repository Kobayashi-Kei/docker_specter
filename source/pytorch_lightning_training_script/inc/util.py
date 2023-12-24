import glob
import json
import torch
import os
import shutil
import argparse
import requests
import wandb
from transformers import AutoTokenizer, AutoModel
from inc.MyDataset import MyDataset
from inc.run_labeled import eval_ranking_metrics
from inc.const import label_dict

# Axcellのデータサイズ(基本medium)
size = "medium"

# 用いる観点をリストで入力
labelList = ["title", "bg", "obj", "method", "res"]

# 入力（埋め込む論文アブストラクトデータ）
dirPath = "../dataserver/axcell/" + size
dataPath = dirPath + "/paperDict.json"
labeledAbstPath = dirPath + "/labeledAbst.json"

sscInputPath = dirPath + "/inputForSSC.jsonl"
sscOutputPath = dirPath + "/resultSSC.txt"


def find_model_checkpoint(model_param_path):
    epoch = 1
    files = glob.glob(model_param_path)
    for filePath in files:
        if f"ep-epoch={epoch}" in filePath:
            return filePath
    raise FileNotFoundError("Model checkpoint not found")

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def save_checkpoint(model, optimizer, dirpath, filename):
    # Create the directory if it doesn't exist
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, dirpath+'/'+filename)

def calculate_gradient_norm(model, norm_type=2):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def prepareData(outputName):
    # 出力（文埋め込み）
    outputDirPath = dirPath + "-" + outputName + "/"
    outputEmbLabelDirPath = outputDirPath + "embLabel/"
    if not os.path.exists(outputDirPath):
        os.mkdir(outputDirPath)
    if not os.path.exists(outputEmbLabelDirPath):
        os.mkdir(outputEmbLabelDirPath)

    # データセットをロード
    paperDict = load_data(dataPath)
    save_data(paperDict, outputDirPath + "paperDict.json")

    # 分類されたアブストラクトをロード
    labeledAbstDict = load_data(labeledAbstPath)
    shutil.copy(labeledAbstPath, outputDirPath)

    # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
    for title in labeledAbstDict:
        labeledAbstDict[title]["title"] = title

    # SSCの結果をロード & 整形
    with open(sscInputPath, 'r') as f:
        sscInput = f.readlines()
    with open(sscOutputPath, 'r') as f:
        sscOutput = f.readlines()

    # アブスト分類の結果を読み込んで、扱いやすいように整形
    sscPaperDict = preprocessSSC(sscOutput, sscInput)

    return paperDict, sscPaperDict, outputEmbLabelDirPath

    
def preprocessSSC(sscOutput, sscInput):
    sscPaperDict = {}
    for i, line in enumerate(sscOutput):
        line = json.loads(line)
        label = line[0]
        text_list = line[1]

        inputLine = json.loads(sscInput[i])
        title = inputLine['title']

        for i, text_label_pair in enumerate(text_list):
            if text_label_pair[1] == 'background_label':
                text_list[i][1] = 'bg'
            elif text_label_pair[1] == 'method_label':
                text_list[i][1] = 'method'
            elif text_label_pair[1] == 'objective_label':
                text_list[i][1] = 'obj'
            elif text_label_pair[1] == 'result_label':
                text_list[i][1] = 'res'
            elif text_label_pair[1] == 'other_label':
                text_list[i][1] = 'other'

        sscPaperDict[title] = text_list
    
    return sscPaperDict

def find_subarray(arr, subarr):
    n = len(arr)
    m = len(subarr)

    # サブ配列が配列に含まれるかどうかを調べる
    for i in range(n - m + 1):
        j = 0
        while j < m and arr[i + j] == subarr[j]:
            j += 1
        if j == m:
            return (i, i + m - 1)

    # サブ配列が配列に含まれない場合は None を返す
    return None

def save_embedding(labeledAbstEmbedding, outputEmbLabelDirPath):
    # ファイル出力
    # labeledAbstSpecter.jsonの名称は評価プログラムでも使っているため変更しない
    with open(outputEmbLabelDirPath + "labeledAbstSpecter.json", 'w') as f:
        json.dump(labeledAbstEmbedding, f, indent=4)


def parse_args(arg_to_scheduler_choices, arg_to_scheduler_metavar):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=None,
                        help='path to the model (if not setting checkpoint)')
    parser.add_argument('--method')
    parser.add_argument('--margin', default=1)
    parser.add_argument('--version', default=0)
    parser.add_argument('--input_dir', default=None,
                        help='optionally provide a directory of the data and train/test/dev files will be automatically detected')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--seed', default=1918, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--test_checkpoint', default=None)
    parser.add_argument('--limit_test_batches', default=1.0, type=float)
    parser.add_argument('--limit_val_batches', default=1.0, type=float)
    parser.add_argument('--val_check_interval', default=1.0, type=float)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    # parser.add_argument("--weight_decay", default=0.0,
    #                     type=float, help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="kwarg passed to DataLoader")
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument('--is_key_transform', default=False, action="store_true")

    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument("--lr_scheduler",
                        default="linear",
                        choices=arg_to_scheduler_choices,
                        metavar=arg_to_scheduler_metavar,
                        type=str,
                        help="Learning rate scheduler")
    parser.add_argument("--data_name", default='axcell')
    parser.add_argument('--gpu', default=0)

    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.input_dir is not None:
        files = glob.glob(args.input_dir + '/*')
        for f in files:
            fname = f.split('/')[-1]
            if 'train' in fname:
                args.train_file = f
            elif 'dev' in fname or 'val' in fname:
                args.dev_file = f
            elif 'test' in fname:
                args.test_file = f
    return args


# LINEに通知する関数
def line_notify(message):
    line_notify_token = 'Jou3ZkH4ajtSTaIWO3POoQvvCJQIdXFyYUaRKlZhHMI'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)


def preprocess_batch(batch, device):
    # 各バッチのデータをGPUに移動
    processed_batch = []
    for item in batch:
        processed_item = {
            'input': {k: v.to(device) for k, v in item['input'].items()},
            'position_label_list': item['position_label_list'].to(device)
        }
        processed_batch.append(processed_item)
    
    return processed_batch


    
def train(model, train_loader, optimizer, scheduler, device, epoch, embedding, is_track_score=True):
    model.train()
    for i, batch in enumerate(train_loader):
        # backwardパスを実行する前に常にそれまでに計算された勾配をクリアする
        # RNNでは勾配の累積は便利だからpytorchは勾配のクリアを自動で行わない
        optimizer.zero_grad()
        batch = preprocess_batch(batch, device)
        loss = model.training_step(batch, i)
        loss.backward()
        optimizer.step()
        scheduler.step()      

        wandb.log({"train_loss": loss.item()})

        
        """
        attention やschedulerが正しく更新されているかのチェック
        """
        # scheduler の学習率
        wandb.log({f'scheduler_lr_batch_{str(epoch)}': scheduler.get_last_lr()[0]})

        # wandbにパラメータの値や勾配をロギング
        wandb.log({f"attnphi_query_grad": calculate_gradient_norm(model.attn_pooling)})
        for name, param in model.attn_pooling.named_parameters():
            wandb.log({f"attnphi_{name}_value": param.data.norm().item()})
            # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        
        wandb.log({f"bert_grad": calculate_gradient_norm(model.bert)})

        # トレーニング後のパラメータの値と比較
        # for name, param in model.attn_pooling.named_parameters():
        #     if not torch.equal(model.attn_pooling_initial_params[name].to(device=device), param.to(device=device)):
        #         print(f"Parameter {name} has changed.")
        #         # exit()
        if is_track_score:
            """
            評価
            """
            if i % 5000 == 0:
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
                paperDict, sscPaperDict, outputEmbLabelDirPath = prepareData(f"{model.hparams.version}-{str(i)}")
                labeledAbstEmbedding = embedding(model, tokenizer, paperDict, sscPaperDict, device)
                save_embedding(labeledAbstEmbedding, outputEmbLabelDirPath)
            
                score_dict = eval_ranking_metrics(f"medium-{model.hparams.version}-{str(i)}", '../dataserver/axcell/')
                wandb.log(score_dict)
                model.train()

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = preprocess_batch(batch, device)
            loss = model.validation_step(batch, i)
            total_loss += loss.item()
            wandb.log({"val_loss": loss})
    average_val_loss = total_loss / len(val_loader)
    wandb.log({"average_val_loss": average_val_loss})

    return average_val_loss


def eval_axcell():
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
    paperDict, sscPaperDict, outputEmbLabelDirPath = prepareData(f"{model.hparams.version}-{str(i)}")
    labeledAbstEmbedding = embedding(model, tokenizer, paperDict, sscPaperDict, device)
    save_embedding(labeledAbstEmbedding, outputEmbLabelDirPath)

    score_dict = eval_ranking_metrics(f"medium-{model.hparams.version}-{str(i)}", '../dataserver/axcell/')
    wandb.log(score_dict)
    model.train()

def embedding(model, tokenizer, paperDict, sscPaperDict, device): 
    # 出力用の辞書
    labeledAbstEmbedding = {}

    my_dataset = MyDataset(None, None, None, label_dict=label_dict)

    # 埋め込み
    for title, paper in paperDict.items():
        labeledAbstEmbedding[title] = {}
        ssc = sscPaperDict[title]

        # Title + [SEP] + Abstract
        title_abs = paper['title'] + \
            tokenizer.sep_token + (paper.get('abstract') or '')
        input = tokenizer(
            title_abs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # 各トークンをBERTに通す
        input = input.to(device)
        output = model.bert(**input)['last_hidden_state']
        # print(output)
        # print(model.bert(**input))

        # print(title_abs)
        # print(ssc)
        _, label_list_for_words = my_dataset.tokenize_with_label(title_abs, ssc, tokenizer)
        label_pooling = model.label_pooling(output, torch.tensor([label_list_for_words]))[0]

        for label in labelList:
            if label_pooling[label] == None or len(label_pooling[label]) == 0:
                labeledAbstEmbedding[title][label] = None
                continue
            # 出力用の辞書に格納
            labeledAbstEmbedding[title][label] = label_pooling[label].tolist()
    
    return labeledAbstEmbedding