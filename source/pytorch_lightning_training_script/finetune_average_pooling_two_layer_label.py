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


    def training_step(self, batch, batch_idx):
        source_label_pooling = self.forward(batch[0]["input"], batch[0]["position_label_list"])
        pos_label_pooling = self.forward(batch[1]["input"], batch[1]["position_label_list"])
        neg_label_pooling = self.forward(batch[2]["input"], batch[2]["position_label_list"])
        
        source_pooling = self.concat_label_tensor(source_label_pooling)
        pos_pooling = self.concat_label_tensor(pos_label_pooling)
        neg_pooling = self.concat_label_tensor(neg_label_pooling)
        
        batch_losses = 0
        for b in range(len(source_pooling)):  # batchsizeの数だけループ (self.hparams.batch_sizeとすると，データが奇数のときindex out Errorが出る)
            entire_loss = self.triple_loss(source_pooling[b], pos_pooling[b], neg_pooling[b])
            batch_losses += entire_loss
        
        loss = batch_losses
        
        return loss

    def validation_step(self, batch, batch_idx):
        source_label_pooling = self.forward(batch[0]["input"], batch[0]["position_label_list"])
        pos_label_pooling = self.forward(batch[1]["input"], batch[1]["position_label_list"])
        neg_label_pooling = self.forward(batch[2]["input"], batch[2]["position_label_list"])
        
        source_pooling = self.concat_label_tensor(source_label_pooling)
        pos_pooling = self.concat_label_tensor(pos_label_pooling)
        neg_pooling = self.concat_label_tensor(neg_label_pooling)
        
        batch_losses = 0
        for b in range(len(source_pooling)):  # batchsizeの数だけループ (self.hparams.batch_sizeとすると，データが奇数のときindex out Errorが出る)
            entire_loss = self.triple_loss(source_pooling[b], pos_pooling[b], neg_pooling[b])
            batch_losses += entire_loss
        
        loss = batch_losses
        
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
            stacked_label_tensor = torch.stack(tensorList)
            retTensor.append(stacked_label_tensor.mean(dim=0))

        return retTensor

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
