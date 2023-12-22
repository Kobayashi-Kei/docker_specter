# transformers, pytorch
from transformers import AutoTokenizer, AutoModel
# transformers, pytorch
from transformers.optimization import (
    Adafactor,
)
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
from inc.qkv_pooling import AttnPhi, make_pad_mask
from inc.SpecterAttnAspect import SpecterAttnAspect

# wandb
import wandb


"""
単語位置出力を観点ごとにaverage poolingして観点埋め込みを生成し，
Finetuning
"""

class Specter(SpecterAttnAspect):
    def calc_label_total_loss(self, source_label_pooling, pos_label_pooling, neg_label_pooling):
        source_pooling = self.average_label_pooling(source_label_pooling)
        pos_pooling = self.average_label_pooling(pos_label_pooling)
        neg_pooling = self.average_label_pooling(neg_label_pooling)

        losses = torch.tensor(0.0, requires_grad=True).to(self.device)
        for b in range(len(source_pooling)):  # batchsizeの数だけループ (self.hparams.batch_sizeとすると，データが奇数のときindex out Errorが出る)
            if not source_pooling[b] == None and not pos_pooling[b] == None and not neg_pooling[b] == None:
                # print(source_pooling[b], pos_pooling[b], neg_pooling[b])
                losses += self.triple_loss(source_pooling[b], pos_pooling[b], neg_pooling[b])
        
        return losses
    
    def average_label_pooling(self, label_pooling):
        retTensor = []
        for b in range(len(label_pooling)):
            tensorList = []
            for label in label_pooling[b]:
                if label_pooling[b][label] == None:
                    continue
                tensorList.append(label_pooling[b][label])
            if (len(tensorList) > 0):
                stacked_label_tensor = torch.stack(tensorList)
                retTensor.append(stacked_label_tensor.mean(dim=0))
            else:
                retTensor.append(None)

        return retTensor


def train_this(model, train_loader, optimizer, scheduler, device, epoch, embedding, is_track_score=True):
    model.train()
    count = 0
    for i, batch in enumerate(train_loader):
        # print(batch)
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
        wandb.log({f"attnphi_title_query_grad": calculate_gradient_norm(model.attn_title)})
        for name, param in model.attn_title.named_parameters():
            wandb.log({f"attnphi_title_{name}_value": param.data.norm().item()})
            # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        wandb.log({f"attnphi_bg_query_grad": calculate_gradient_norm(model.attn_bg)})
        for name, param in model.attn_bg.named_parameters():
            wandb.log({f"attnphi_bg_{name}_value": param.data.norm().item()})
            # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        # wandb.log({f"attnphi_obj_query_grad": calculate_gradient_norm(model.attn_obj)})
        # for name, param in model.attn_obj.named_parameters():
        #     wandb.log({f"attnphi_obj_{name}_value": param.data.norm().item()})
            # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        wandb.log({f"attnphi_method_query_grad": calculate_gradient_norm(model.attn_method)})
        for name, param in model.attn_method.named_parameters():
            wandb.log({f"attnphi_method_{name}_value": param.data.norm().item()})
            # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        wandb.log({f"attnphi_res_query_grad": calculate_gradient_norm(model.attn_res)})
        for name, param in model.attn_res.named_parameters():
            wandb.log({f"attnphi_res_{name}_value": param.data.norm().item()})
            # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        
        wandb.log({f"bert_grad": calculate_gradient_norm(model.bert)})

        count += 1
        wandb.log({"count": count})

        # トレーニング後のパラメータの値と比較
        # for name, param in model.attn_pooling.named_parameters():
        #     if not torch.equal(model.attn_pooling_initial_params[name].to(device=self.device), param.to(device=self.device)):
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
                labeledAbstEmbedding = embedding(model, tokenizer, paperDict, sscPaperDict)
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
        wandb.init(project=args.version, config=args)

        # 学習のメインプログラム
        model = Specter(args).to(args.device)
        optimizer, scheduler = model.configure_optimizers()
        train_loader = model._get_loader("train", args.data_name)
        val_loader = model._get_loader("dev", args.data_name)

        # val_loss = validate(model, val_loader, device)
        # print(f"Init, Val Loss: {val_loss}")
        for epoch in range(args.num_epochs):
            # train(model, train_loader, optimizer, scheduler, device, epoch, embedding)
            train_this(model, train_loader, optimizer, scheduler, device, epoch, embedding)
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
