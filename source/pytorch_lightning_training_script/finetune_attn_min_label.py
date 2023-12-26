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
単語位置出力を観点ごとにaverage poolingして観点埋め込みを生成し，
Finetuning
"""
class Specter(SpecterOrigin):
    def __init__(self, init_args={}):
        super().__init__(init_args)

    def calc_label_total_loss(self, source_label_pooling, pos_label_pooling, neg_label_pooling):
        losses = torch.tensor(0.0, requires_grad=True).to(device=self.hparams.device)
        valid_batch = 0
        for b in range(len(source_label_pooling)):  # batchsizeの数だけループ
            label_loss_list = []
            for label in label_dict:
                if not source_label_pooling[b][label] == None and not pos_label_pooling[b][label] == None and not neg_label_pooling[b][label] == None:
                    label_loss_list.append(self.triple_loss(
                        source_label_pooling[b][label], pos_label_pooling[b][label], neg_label_pooling[b][label]))

            if len(label_loss_list) > 0:
                label_loss_list = torch.stack(label_loss_list)
                # torch.sort(label_loss_list, 0) は[0]に並び替えたtensor, [1]にそのindexのtensorが返る
                losses += torch.sort(label_loss_list, 0)[0][0]
                valid_batch += 1

        if valid_batch > 0:
            return losses / valid_batch
        else:
            return losses


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
