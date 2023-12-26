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

    def calc_label_total_loss(self, source_label_pooling, pos_label_pooling, neg_label_pooling):
        losses = torch.tensor(0.0, requires_grad=True).to(device=self.hparams.device)
        valid_batch = 0
        for b in range(len(source_label_pooling)): # batchsizeの数だけループ
            label_loss = 0
            valid_label_list = []
            for label in label_dict:
                if not source_label_pooling[b][label] == None and not pos_label_pooling[b][label] == None and not neg_label_pooling[b][label] == None:
                    valid_label_list.append(label)
                    # debug print
                    print("--" , label)
                    label_loss += self.triple_loss(source_label_pooling[b][label], pos_label_pooling[b][label], neg_label_pooling[b][label])

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
            train(model, tokenizer, train_loader, optimizer, scheduler, args.device, epoch, embedding)
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
