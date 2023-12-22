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
        batch_loss = torch.tensor(0.0, requires_grad=True).to(device=self.device)
        for b in range(len(source_label_pooling)):  # batchsizeの数だけループ
            entire_source_emb = torch.zeros(self.bert.config.hidden_size).to(device=self.device)
            entire_pos_emb = torch.zeros(self.bert.config.hidden_size).to(device=self.device)
            entire_neg_emb = torch.zeros(self.bert.config.hidden_size).to(device=self.device)
            for label in label_dict:
                """
                tripleに存在する観点のみ加えるパターン
                """
                # if not source_label_pooling[b][label] == None and not pos_label_pooling[b][label] == None and not neg_label_pooling[b][label] == None:
                #     entire_source_emb += source_label_pooling[b][label]
                #     entire_pos_emb += pos_label_pooling[b][label]
                #     entire_neg_emb += neg_label_pooling[b][label]
                    
                """
                tripleすべてに存在しなくても足すパターン
                """
                if not source_label_pooling[b][label] == None:
                    entire_source_emb += source_label_pooling[b][label]
                if not pos_label_pooling[b][label] == None:
                    entire_pos_emb += pos_label_pooling[b][label]
                if not neg_label_pooling[b][label] == None:
                    entire_neg_emb += neg_label_pooling[b][label]

            if not torch.equal(entire_source_emb, torch.zeros(self.bert.config.hidden_size).to(device=self.device)) and not torch.equal(entire_pos_emb, torch.zeros(self.bert.config.hidden_size).to(device=self.device)) and not torch.equal(entire_neg_emb, torch.zeros(self.bert.config.hidden_size).to(device=self.device)):
                batch_loss += self.triple_loss(
                        entire_source_emb, entire_pos_emb, entire_neg_emb)
            else:
                batch_loss += torch.tensor(0.0, requires_grad=True)
            
            # print(batch_loss)

        return batch_loss
    

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
