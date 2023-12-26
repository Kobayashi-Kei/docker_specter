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
import json, codecs

# inc
from inc.util import save_checkpoint,  calculate_gradient_norm
from inc.util import prepareData, save_embedding, parse_args, line_notify, preprocess_batch

from inc.const import label_dict, num_label_dict
from inc.const import arg_to_scheduler, arg_to_scheduler_choices, arg_to_scheduler_metavar
from inc.util import validate, embedding, embedding_axcell
from inc.const import tokenizer_name
from inc.qkv_pooling import AttnPhi, make_pad_mask
from inc.SpecterAttnAspect import SpecterAttnAspect
from inc.util import initialize_environment, calc_gpus

from inc.run_labeled import eval_log_ranking_metrics
from inc.csf_util import eval_log_CSFCube
from inc.eval_similar_label import eval_log_similar_label

# wandb
import wandb


"""
単語位置出力を観点ごとにaverage poolingして観点埋め込みを生成し，
Finetuning
"""

class Specter(SpecterAttnAspect):
    def calc_label_total_loss(self, source_label_pooling, pos_label_pooling, neg_label_pooling):
        losses = torch.tensor(0.0, requires_grad=True).to(device=self.hparams.device)
        valid_batch = 0
        for b in range(len(source_label_pooling)):  # batchsizeの数だけループ
            label_loss = torch.tensor(0.0, requires_grad=True).to(device=self.hparams.device)
            valid_label_list = []
            for label in label_dict:
                if label=='other':
                    continue
                if not source_label_pooling[b][label] == None and not pos_label_pooling[b][label] == None and not neg_label_pooling[b][label] == None:
                    valid_label_list.append(label)
                    loss = self.triple_loss(
                        source_label_pooling[b][label], pos_label_pooling[b][label], neg_label_pooling[b][label])
                    # print(self.triple_loss(
                    #     source_label_pooling[b][label], pos_label_pooling[b][label], neg_label_pooling[b][label]))
                    # print(torch.tanh(loss))
                    if self.hparams.tanh:
                        loss = torch.tanh(loss*self.hparams.tanh_coefficient)
                    label_loss += loss


            if len(valid_label_list) > 0:
                losses += label_loss / len(valid_label_list)
                valid_batch += 1

        if valid_batch > 0:
            return losses / valid_batch
        else:
            return losses



def train_this(model, tokenizer, train_loader, optimizer, scheduler, device, epoch, embedding, is_track_score=True):
    model.train()
    count = 0
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

        # 評価
        if is_track_score:
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
