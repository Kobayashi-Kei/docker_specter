# transformers, pytorch
from transformers.optimization import (
    Adafactor,
)
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW

import torch

# basic python packages
import random
import numpy as np
import os
import traceback

# inc
from inc.util import save_checkpoint,  calculate_gradient_norm
from inc.util import prepareData, save_embedding, parse_args, line_notify, preprocess_batch
from inc.run_labeled import eval_ranking_metrics
from inc.SpecterOrigin import SpecterOrigin
from inc.const import label_dict, num_label_dict
from inc.const import arg_to_scheduler, arg_to_scheduler_choices, arg_to_scheduler_metavar
from inc.util import train, validate, embedding
from inc.const import tokenizer_name


# wandb
import wandb

"""
自分で定義するデータセット
"""
class Specter(SpecterOrigin):
    def __init__(self, init_args={}):
        super().__init__(init_args)
        del self.attn_pooling

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        # old
        # no_decay = ["bias", "LayerNorm.weight"]
        # model_parameters = self.named_parameters()
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model_parameters if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        #     {
        #         "params": [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.hparams.weight_decay,
        #         # "weight_decay": 0.0,
        #     },
        # ]
        
        # new!
        # weight decay 重み減衰：過学習を減らすためにparameterの自由度を減らす
        optimizer_grouped_parameters = [
            {
                "params": [],
                "weight_decay": 0.0,
            },
            {
                "params": [],
                "weight_decay": self.hparams.weight_decay,
                # "weight_decay": 0.0,
            },
        ]
        bert_params = self.make_parameter_group(self.bert)
        optimizer_grouped_parameters[0]['params'].extend(bert_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(bert_params['decay'])

        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon # type: ignore # type: ignore
            )

        self.opt = optimizer
        
        # get_lr_scheduler内でself.optを渡す
        scheduler = self.get_lr_scheduler()

        return optimizer, scheduler


    def training_step(self, batch, batch_idx):
        source_label_pooling = self.forward(batch[0]["input"], batch[0]["position_label_list"])
        pos_label_pooling = self.forward(batch[1]["input"], batch[1]["position_label_list"])
        neg_label_pooling = self.forward(batch[2]["input"], batch[2]["position_label_list"])
        
        source_pooling = self.concat_label_tensor(source_label_pooling)
        pos_pooling = self.concat_label_tensor(pos_label_pooling)
        neg_pooling = self.concat_label_tensor(neg_label_pooling)
        
        losses = torch.tensor(0.0, requires_grad=True).to(device=self.hparams.device)
        valid_batch = 0
        for b in range(len(source_pooling)):  # batchsizeの数だけループ (self.hparams.batch_sizeとすると，データが奇数のときindex out Errorが出る)
            if not source_pooling[b] == None and not pos_pooling[b] == None and not neg_pooling[b] == None:
                losses += self.triple_loss(source_pooling[b], pos_pooling[b], neg_pooling[b])
                valid_batch += 1
        
        if valid_batch > 0:
            return losses / valid_batch
        else:
            return losses

    def validation_step(self, batch, batch_idx):
        source_label_pooling = self.forward(batch[0]["input"], batch[0]["position_label_list"])
        pos_label_pooling = self.forward(batch[1]["input"], batch[1]["position_label_list"])
        neg_label_pooling = self.forward(batch[2]["input"], batch[2]["position_label_list"])
        
        source_pooling = self.concat_label_tensor(source_label_pooling)
        pos_pooling = self.concat_label_tensor(pos_label_pooling)
        neg_pooling = self.concat_label_tensor(neg_label_pooling)
        
        losses = torch.tensor(0.0, requires_grad=True).to(device=self.hparams.device)
        valid_batch = 0
        for b in range(len(source_pooling)):  # batchsizeの数だけループ (self.hparams.batch_sizeとすると，データが奇数のときindex out Errorが出る)
            if not source_pooling[b] == None and not pos_pooling[b] == None and not neg_pooling[b] == None:
                losses += self.triple_loss(source_pooling[b], pos_pooling[b], neg_pooling[b])
                valid_batch += 1
        
        if valid_batch > 0:
            return losses / valid_batch
        else:
            return losses

    def label_pooling(self, batch_last_hidden_state, batch_position_label_list):
        batch_size = batch_last_hidden_state.size(0)
        
        batch_label_pooling = []
        # print()
        # print("--batch_last_hidden_state--")
        # print(batch_last_hidden_state)
        # print("--batch_position_label_list--")
        # print(batch_position_label_list)

        for b in range(batch_size):
            label_pooling = {}
            last_hidden_state = batch_last_hidden_state[b]
            position_label_list = batch_position_label_list[b]
            # print(position_label_list)
            label_last_hidden_state = {}
            for label in label_dict:
                label_last_hidden_state[label] = []

            # 各単語のlast_hidden_stateを観点ごとのリストに格納
            # print("\n", batch_position_label_list[b])
            for i, label_tensor in enumerate(position_label_list):
                label_number = str(label_tensor.item()) # tensor -> str
                # [CLS]と[SEP]の位置はskip
                if label_number == "-1":
                    continue
                # 文末以降は'0'埋めだから終わる
                if label_number == '0':
                    break
                # ex. "1" -> "bg"
                label = num_label_dict[label_number]
                # print(label)
                label_last_hidden_state[label].append(
                    last_hidden_state[i])

            # average poolingで集約
            for label in label_last_hidden_state:
                # ２次元のテンソルに変換
                if len(label_last_hidden_state[label]) > 0:
                    label_last_hidden_state_tensor = torch.stack(
                        label_last_hidden_state[label])
                    # print(label_last_hidden_state_tensor)
                    label_pooling[label] = label_last_hidden_state_tensor.mean(
                        dim=0)
                else:
                    label_pooling[label] = None

            batch_label_pooling.append(label_pooling)

        # print(batch_label_pooling)
        # print(len(batch_label_pooling))
        # print(len(batch_label_pooling[0]))
        # for batch in range(len(batch_label_pooling)):
        #     for label in label_dict:
        #         print(label)
        #         print(batch_label_pooling[batch][label])
        #         if label == "obj":
        #             exit()
        return batch_label_pooling

    def concat_label_tensor(self, label_tensor):
        retTensor = []
        for b in range(len(label_tensor)):
            tensorList = []
            for label in label_tensor[b]:
                if label_tensor[b][label] == None:
                    continue
                tensorList.append(label_tensor[b][label])
            if len(tensorList) > 0:
                stacked_label_tensor = torch.stack(tensorList)
                retTensor.append(stacked_label_tensor.mean(dim=0))
            else:
                retTensor.append(None)

        return retTensor
    
def train_this(model, train_loader, optimizer, scheduler, device, epoch, embedding, is_track_score=True):
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
                

def main():
    try:
        # 引数の取得，設定
        args = parse_args(arg_to_scheduler_choices, arg_to_scheduler_metavar)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.num_workers == 0:
            print("num_workers cannot be less than 1")
            return

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if ',' in args.gpus:
            args.gpus = list(map(int, args.gpus.split(',')))
            args.total_gpus = len(args.gpus)
        else:
            args.gpus = int(args.gpus)
            args.total_gpus = args.gpus

        save_dir = f'/workspace/dataserver/model_outputs/specter/{args.version}/'

        # wandbの初期化
        wandb.init(project=args.version, config=args) # type: ignore # type: ignore # type: ignore # type: ignore # type: ignore

        # 学習のメインプログラム
        model = Specter(args).to(args.device)
        optimizer, scheduler = model.configure_optimizers()
        train_loader = model._get_loader("train", args.data_name)
        val_loader = model._get_loader("dev", args.data_name)

        # val_loss = validate(model, val_loader, args.device)
        # print(f"Init, Val Loss: {val_loss}")
        for epoch in range(args.num_epochs):
            train_this(model, train_loader, optimizer, scheduler, args.device, epoch, embedding)
            val_loss = validate(model, val_loader, args.device)
            print(f"Epoch {epoch}, Val Loss: {val_loss}")
            save_checkpoint(model, optimizer,save_dir, f"ep-epoch={epoch}.pth.tar")
        
        # 終了処理（LINE通知，casheとロギングの終了）
        line_notify("172.21.64.47:" + os.path.basename(__file__) + "が終了")

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        paperDict, sscPaperDict, outputEmbLabelDirPath = prepareData(args.version)
        labeledAbstEmbedding = embedding(model, tokenizer, paperDict, sscPaperDict, args.device)
        save_embedding(labeledAbstEmbedding, outputEmbLabelDirPath)
        
        score_dict = eval_ranking_metrics(f"medium-{args.version}", '../dataserver/axcell/')

        line_notify(args.version + ': ' + '{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(
            score_dict['mrr'], 
            score_dict['map@10'], 
            score_dict['map@20'], 
            score_dict['recall@10'], 
            score_dict['recall@20']
        ))
        wandb.log(score_dict)
        
        wandb.finish()
        torch.cuda.empty_cache()

    except Exception as e:
        print(traceback.format_exc())
        message = "172.21.65.47: Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        line_notify(message)


if __name__ == '__main__':
    main()
