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
from inc.util import validate, embedding, embedding_axcell
from inc.const import tokenizer_name
from inc.util import initialize_environment, calc_gpus

from inc.run_labeled import eval_log_ranking_metrics
from inc.csf_util import eval_log_CSFCube
from inc.eval_similar_label import eval_log_similar_label

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
        # self.lstm_obj = torch.nn.LSTM(input_size=self.bert.config.hidden_size,
        #                     hidden_size=self.bert.config.hidden_size, batch_first=True)
        self.lstm_method = torch.nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=self.bert.config.hidden_size, batch_first=True)
        self.lstm_res = torch.nn.LSTM(input_size=self.bert.config.hidden_size,
                             hidden_size=self.bert.config.hidden_size, batch_first=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

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

        lstm_params = self.make_parameter_group(self.lstm_title)
        optimizer_grouped_parameters[0]['params'].extend(lstm_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(lstm_params['decay'])
        lstm_params = self.make_parameter_group(self.lstm_bg)
        optimizer_grouped_parameters[0]['params'].extend(lstm_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(lstm_params['decay'])
        # lstm_params = self.make_parameter_group(self.lstm_obj)
        # optimizer_grouped_parameters[0]['params'].extend(lstm_params['no_decay'])
        # optimizer_grouped_parameters[1]['params'].extend(lstm_params['decay'])
        lstm_params = self.make_parameter_group(self.lstm_method)
        optimizer_grouped_parameters[0]['params'].extend(lstm_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(lstm_params['decay'])
        lstm_params = self.make_parameter_group(self.lstm_res)
        optimizer_grouped_parameters[0]['params'].extend(lstm_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(lstm_params['decay'])
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
    
    def calc_label_total_loss(self, source_label_pooling, pos_label_pooling, neg_label_pooling):
        losses = torch.tensor(0.0, requires_grad=True).to(device=self.hparams.device)
        valid_batch = 0
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
                losses += label_loss / len(valid_label_list)
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
                    elif label == 'other':
                        label_pooling[label] = None
                        continue

                    label_pooling[label] = output[-1] #type: ignore
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

def train_this(model, tokenizer, train_loader, optimizer, scheduler, device, epoch, embedding, is_track_score=True):
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

        if is_track_score:
            """
            評価
            """
            if i % 10000 == 0:
                embedding_axcell(model, tokenizer, f"{model.hparams.version}-{str(i)}", device)
                eval_log_ranking_metrics(f"medium-{model.hparams.version}-{str(i)}", '../dataserver/axcell/')
                eval_log_similar_label(f"medium-{model.hparams.version}-{str(i)}", "../dataserver/axcell/")
                eval_log_CSFCube(model, tokenizer, device, f"{model.hparams.version}-{str(i)}")
                model.train()

    
def main():
    try:
        # 引数の取得，設定
        args = parse_args(arg_to_scheduler_choices, arg_to_scheduler_metavar)
        initialize_environment(args)
        args = calc_gpus(args)

        save_dir = f'/workspace/dataserver/model_outputs/specter/{args.version}/'

        # wandbの初期化
        wandb.init(project=args.version, config=args) # type: ignore

        # 学習のメインプログラム
        model = Specter(args).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        optimizer, scheduler = model.configure_optimizers()
        train_loader = model._get_loader("train", args.data_name)
        val_loader = model._get_loader("dev", args.data_name)

        # val_loss = validate(model, val_loader, args.device)
        # print(f"Init, Val Loss: {val_loss}")
        for epoch in range(args.num_epochs):
            train_this(model, tokenizer, train_loader, optimizer, scheduler, args.device, epoch, embedding)
            val_loss = validate(model, val_loader, args.device)
            print(f"Epoch {epoch}, Val Loss: {val_loss}")
            save_checkpoint(model, optimizer,save_dir, f"ep-epoch={epoch}.pth.tar")
        

        # 評価
        embedding_axcell(model, tokenizer, args.version, args.device)
        eval_log_ranking_metrics(f"medium-{args.version}", '../dataserver/axcell/')
        eval_log_similar_label(f"medium-{args.version}", "../dataserver/axcell/")
        eval_log_CSFCube(model, tokenizer, args.device, args.version)


        # 終了処理（LINE通知，casheとロギングの終了）
        wandb.finish()
        torch.cuda.empty_cache()
        line_notify(os.path.basename(__file__) + "が終了")
        

    except Exception as e:
        print(traceback.format_exc())
        message = "172.21.65.47: Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        line_notify(message)


if __name__ == '__main__':
    main()
