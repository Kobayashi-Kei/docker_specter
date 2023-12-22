# transformers, pytorch
from transformers import AutoTokenizer, AutoModel

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
        self.lstm_title= torch.nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=self.bert.config.hidden_size, batch_first=True)
        self.lstm_bg = torch.nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=self.bert.config.hidden_size, batch_first=True)
        self.lstm_obj = torch.nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=self.bert.config.hidden_size, batch_first=True)
        self.lstm_method = torch.nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=self.bert.config.hidden_size, batch_first=True)
        self.lstm_res = torch.nn.LSTM(input_size=self.bert.config.hidden_size,
                             hidden_size=self.bert.config.hidden_size, batch_first=True)

    def calc_label_total_loss(self, source_label_pooling, pos_label_pooling, neg_label_pooling):
        batch_label_loss = 0
        label_loss_calculated_count = 0
        for b in range(len(source_label_pooling)):  # batchsizeの数だけループ
            label_loss = 0
            valid_label_list = []
            for label in label_dict:
                if not source_label_pooling[b][label] == None and not pos_label_pooling[b][label] == None and not neg_label_pooling[b][label] == None:
                    valid_label_list.append(label)
                    # debug print
                    # print("--", label)
                    label_loss += self.triple_loss(
                        source_label_pooling[b][label], pos_label_pooling[b][label], neg_label_pooling[b][label])

            if len(valid_label_list) > 0:
                batch_label_loss += label_loss/len(valid_label_list)
                label_loss_calculated_count += 1

        """
        一致する観点がなければlossは0
        """
        if label_loss_calculated_count > 0:
            loss = batch_label_loss / label_loss_calculated_count
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        return loss

    def label_pooling(self, batch_last_hidden_state, batch_position_label_list):
        batch_size = batch_last_hidden_state.size(0)
        label_dict = self.hparams.label_dict
        num_label_dict = self.hparams.num_label_dict
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
                    # outputに各単語に対応する隠れ層の出力（単語数×次元数のtensor）, _ に最後の単語に対応する隠れ層の出力とCtのタプル
                    # つまり，outputの最後の要素を予測に利用する
                    # 参考: https://qiita.com/m__k/items/841950a57a0d7ff05506#%E3%83%A2%E3%83%87%E3%83%AB%E5%AE%9A%E7%BE%A9
                    if label == 'bg':
                        output, _ = self.lstm_bg(
                            label_last_hidden_state_tensor, None)
                    elif label == 'obj':
                        output, _ = self.lstm_obj(
                            label_last_hidden_state_tensor, None)
                    elif label == 'method':
                        output, _ = self.lstm_title(
                            label_last_hidden_state_tensor, None)
                    elif label == 'res':
                        output, _ = self.lstm_res(
                            label_last_hidden_state_tensor, None)
                    elif label == 'title':
                        output, _ = self.lstm_title(
                            label_last_hidden_state_tensor, None)
                    
                    label_pooling[label] = output[-1]
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
        wandb.init(project=args.version, config=args)

        # 学習のメインプログラム
        model = Specter(args).to(args.device)
        optimizer, scheduler = model.configure_optimizers()
        train_loader = model._get_loader("train")
        val_loader = model._get_loader("dev")

        val_loss = validate(model, val_loader, device)
        print(f"Init, Val Loss: {val_loss}")
        for epoch in range(args.num_epochs):
            train(model, train_loader, optimizer, scheduler, device, epoch, embedding)
            val_loss = validate(model, val_loader, device)
            print(f"Epoch {epoch}, Val Loss: {val_loss}")
            save_checkpoint(model, optimizer,save_dir, f"ep-epoch={epoch}.pth.tar")
        
        wandb.finish()

        # 終了処理（LINE通知，casheとロギングの終了）
        line_notify("172.21.64.47:" + os.path.basename(__file__) + "が終了")

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        paperDict, sscPaperDict, outputEmbLabelDirPath = prepareData(args.version)
        labeledAbstEmbedding = embedding(model, tokenizer, paperDict, sscPaperDict)
        save_embedding(labeledAbstEmbedding, outputEmbLabelDirPath)
        
        score_dict = eval_ranking_metrics(f"medium-{args.version}", '../dataserver/axcell/')

        line_notify(args.version + ': ' + '{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(
            score_dict['mrr'], 
            score_dict['map@10'], 
            score_dict['map@20'], 
            score_dict['recall@10'], 
            score_dict['recall@20']
        ))

        torch.cuda.empty_cache()
        

    except Exception as e:
        print(traceback.format_exc())
        message = "172.21.65.47: Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        line_notify(message)


if __name__ == '__main__':
    main()
