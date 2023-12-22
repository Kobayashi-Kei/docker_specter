# transformers, pytorch
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

# basic python packages
import json
import pickle
import argparse
from argparse import Namespace
import glob
import random
import numpy as np
import requests
import logging
import os
import traceback
import matplotlib.pyplot as plt
import re

# inc
from inc.MyDataset import MyDataset
from inc.TripletLoss import TripletLoss
from inc.qkv_pooling import AttnPhi, make_pad_mask

# wandb
from pytorch_lightning.loggers import WandbLogger



"""
単語位置出力を観点ごとにaverage poolingして観点埋め込みを生成し，
Finetuning
"""


logger = logging.getLogger(__name__)

# Globe constants
training_size = 684100
# validation_size = 145375

# log_every_n_steps how frequently pytorch lightning logs.
# By default, Lightning logs every 50 rows, or 50 training steps.
log_every_n_steps = 1

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class Specter(pl.LightningModule):
    def __init__(self, init_args):
        super().__init__()
        self.save_hyperparameters()
        if isinstance(init_args, dict):
            # for loading the checkpoint, pl passes a dict (hparams are saved as dict)
            init_args = Namespace(**init_args)
        checkpoint_path = init_args.checkpoint_path
        logger.info(f'loading model from checkpoint: {checkpoint_path}')

        self.hparams = init_args

        # SciBERTを初期値
        self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        
        # SPECTERを初期値とする場合
        # self.bert = AutoModel.from_pretrained("allenai/specter")
        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

        # Average Pooling Specterを初期値
        # modelCheckpoint = '/workspace/dataserver/model_outputs/specter/pretrain_average_pooling/checkpoints/ep-epoch=1_avg_val_loss-avg_val_loss=0.260-v0.ckpt'
        # self.bert = Specter.load_from_checkpoint(modelCheckpoint).bert
        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

        self.tokenizer.model_max_length = self.bert.config.max_position_embeddings
        self.hparams.seqlen = self.bert.config.max_position_embeddings
        self.triple_loss = TripletLoss(margin=float(init_args.margin))

        self.attn_pooling = AttnPhi(self.bert.config.hidden_size)
        # トレーニング前のパラメータの値を保存
        self.attn_pooling_initial_params = {n: p.clone() for n, p in self.attn_pooling.named_parameters()}
        
        # cosine類似度も試す
        # self.triple_loss = TripletLoss(
        #     distance='cosine', margin=float(init_args.margin))
        
        # number of training instances
        self.training_size = None
        # number of testing instances
        self.validation_size = None
        # number of test instances
        self.test_size = None
        # This is a dictionary to save the embeddings for source papers in test step.
        self.embedding_output = {}

        self.debug = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        source_embedding = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return source_embedding

    """
    このメソッドでallennlp用のデータをロードして、トークナイズまで行う（tokentype_id, attention_maskなど)
    -> つまりこのメソッドに関わる箇所を書き換えればいい。
    """
    def _get_loader(self, split):
        # Axcell データ
        path = "/workspace/dataserver/axcell/large/specter/paper/triple-" + split + ".json"
        with open(path, 'r') as f:
            data = json.load(f)

        path = "/workspace/dataserver/axcell/large/paperDict.json"
        with open(path, 'r') as f:
            paper_dict = json.load(f)

        path = "/workspace/dataserver/axcell/large/result_ssc.json"
        with open(path, 'r') as f:
            ssc_result_dict = json.load(f)

        # SPECTER データ
        # path = "/workspace/dataserver/specterData/label/" + split + "/specter_triple-" + split + ".json"
        # with open(path, 'r') as f:
        #     data = json.load(f)

        # path = "/workspace/dataserver/specterData/label/" + split + "/specter_paperDict.json"
        # with open(path, 'r') as f:
        #     paper_dict = json.load(f)

        # path = "/workspace/dataserver/specterData/label/" + split + "/result_ssc.json"
        # with open(path, 'r') as f:
        #     ssc_result_dict = json.load(f)

        dataset = MyDataset(data, paper_dict, ssc_result_dict,
                         self.hparams.label_dict, self.tokenizer)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                            shuffle=False, pin_memory=False)

        # print(loader)
        return loader

    """
    これはよくわからん、絶対に呼び出されるやつ？
    """

    def setup(self, mode):
        self.train_loader = self._get_loader("train")

    """
    以下はすべてPytorch Lightningの指定のメソッド
    そのためこのファイル内には呼び出している箇所は無い。
    """
    """
    allennlp用のデータを読み取り、変換する
    """

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        self.val_dataloader_obj = self._get_loader('dev')
        return self.val_dataloader_obj

    def test_dataloader(self):
        return self._get_loader('test')

    """
    学習の設定等に関わるメソッド群
    """
    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(
            1, self.hparams.total_gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.batch_size * \
            self.hparams.grad_accum * num_devices
        # dataset_size = len(self.train_loader.dataset)
        """The size of the training data need to be coded with more accurate number"""
        dataset_size = len(self._get_loader("train"))
        return (dataset_size / effective_batch_size) * self.hparams.num_epochs

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler,
                     "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]

        # Combine parameters of BERT and LSTM
        # model_parameters = list(self.bert.named_parameters()) + \
        #     list(self.attn_pooling.named_parameters())
        # model_parameters = self.named_parameters()
        model_parameters = self.bert.named_parameters()
        # print(model_parameters)
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # source_embedding = self.bert(**batch[0])[1] # [1]はpooler_outputのこと　https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
        # pos_embedding = self.bert(**batch[1])[1]
        # neg_embedding = self.bert(**batch[2])[1]
        source_embedding = self.forward(**batch[0]["input"])['last_hidden_state']
        pos_embedding = self.forward(**batch[1]["input"])['last_hidden_state']
        neg_embedding = self.forward(**batch[2]["input"])['last_hidden_state']
        # last hidden stateのtensor形状は
        # ( バッチサイズ, 系列長(512), 次元数(768) )

        """
        観点のロス計算
        """
        source_label_pooling = self.label_pooling(
            source_embedding, batch[0]["position_label_list"])
        pos_label_pooling = self.label_pooling(
            pos_embedding, batch[1]["position_label_list"])
        neg_label_pooling = self.label_pooling(
            neg_embedding, batch[2]["position_label_list"])
        
        batch_label_loss = 0
        label_loss_calculated_count = 0
        for b in range(len(source_label_pooling)):  # batchsizeの数だけループ
            label_loss = 0
            valid_label_list = []
            for label in self.hparams.label_dict:
                if not source_label_pooling[b][label] == None and not pos_label_pooling[b][label] == None and not neg_label_pooling[b][label] == None:
                    valid_label_list.append(label)
                    # debug print
                    print("--", label)
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
        
        if self.debug:
            return {"loss": loss}

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        self.log('train_loss', loss, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)
        self.log('rate', lr_scheduler.get_last_lr()
                 [-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # wandbにパラメータの値や勾配をロギング
        for name, param in self.attn_pooling.named_parameters():
            if param.grad is not None:
                self.log(f"{name}_grad", param.grad.norm().item())
            self.log(f"{name}_value", param.data.norm().item())
        
        # パラメータを出力
        # for name, param in self.attn_pooling.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

        # トレーニング後のパラメータの値と比較
        for name, param in self.attn_pooling.named_parameters():
            if not torch.equal(self.attn_pooling_initial_params[name].to(device="cuda:0"), param.to(device="cuda:0")):
                print(f"Parameter {name} has changed.")
                exit()
            
            
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # source_embedding = self.bert(**batch[0])[1] # [1]はpooler_outputのこと　https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
        # pos_embedding = self.bert(**batch[1])[1]
        # neg_embedding = self.bert(**batch[2])[1]
        source_embedding = self.forward(**batch[0]["input"])['last_hidden_state']
        pos_embedding = self.forward(**batch[1]["input"])['last_hidden_state']
        neg_embedding = self.forward(**batch[2]["input"])['last_hidden_state']
        # last hidden stateのtensor形状は
        # ( バッチサイズ, 系列長(512), 次元数(768) )

        """
        観点のロス計算
        """
        source_label_pooling = self.label_pooling(
            source_embedding, batch[0]["position_label_list"])
        pos_label_pooling = self.label_pooling(
            pos_embedding, batch[1]["position_label_list"])
        neg_label_pooling = self.label_pooling(
            neg_embedding, batch[2]["position_label_list"])
        
        batch_label_loss = 0
        label_loss_calculated_count = 0
        for b in range(len(source_label_pooling)):  # batchsizeの数だけループ
            label_loss = 0
            valid_label_list = []
            for label in self.hparams.label_dict:
                if not source_label_pooling[b][label] == None and not pos_label_pooling[b][label] == None and not neg_label_pooling[b][label] == None:
                    valid_label_list.append(label)
                    # debug print
                    print("--", label)
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

        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return {'val_loss': loss}

    def _eval_end(self, outputs) -> tuple:
        # avg_loss = torch.stack([x['val_loss'].to('cuda:0') for x in outputs]).mean()
        # なぜか次のエラーが出るため，RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument tensors in method wrapper_cat)
        avg_loss = torch.stack([x['val_loss'].to('cuda:0') for x in outputs]).mean()
        if self.trainer.use_ddp:
            torch.distributed.all_reduce(
                avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size
        results = {"avg_val_loss": avg_loss}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach().cpu().item()
        return results

    def validation_epoch_end(self, outputs: list) -> dict:
        ret = self._eval_end(outputs)

        self.log('avg_val_loss', ret["avg_val_loss"],
                 on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs: list):
        # convert the dictionary of {id1:embedding1, id2:embedding2, ...} to a
        # list of dictionaries [{'id':'id1', 'embedding': 'embedding1'},{'id':'id2', 'embedding': 'embedding2'}, ...]
        embedding_output_list = [{'id': key, 'embedding': value.detach().cpu().numpy().tolist()}
                                 for key, value in self.embedding_output.items()]

        with open(self.hparams.save_dir+'/embedding_result.jsonl', 'w') as fp:
            fp.write('\n'.join(json.dumps(i) for i in embedding_output_list))

    def test_step(self, batch, batch_nb):
        source_embedding = self.bert(**batch[0])[1]
        source_paper_id = batch[1]

        batch_embedding_output = dict(zip(source_paper_id, source_embedding))

        # .update() will automatically remove duplicates.
        self.embedding_output.update(batch_embedding_output)
        # return self.validation_step(batch, batch_nb)

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
                    # Attnpoolingの対象が1観点（1文章）の場合はpadding_maskは実質必要無いが，元のコードを変更しないため，全てmaskしない設定で行う
                    # 必要な場合は以下のように単語数が異なる複数の文章を同時にpoolingする時
                    # [false false true true]
                    # [false false false false]
                    lengths = torch.tensor([label_last_hidden_state_tensor.size(0)])
                    label_last_hidden_state_padding_mask = make_pad_mask(lengths).to(device="cuda:0")
                    # print(label_last_hidden_state_tensor.unsqueeze(0).size())
                    # print(label_last_hidden_state_padding_mask)
                    label_pooling[label] = self.attn_pooling(label_last_hidden_state_tensor.unsqueeze(0), label_last_hidden_state_padding_mask)[0]
                    # print(label_pooling[label].size())
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
            


def parse_args():
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
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="kwarg passed to DataLoader")
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument('--save_dir', required=True)

    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument("--lr_scheduler",
                        default="linear",
                        choices=arg_to_scheduler_choices,
                        metavar=arg_to_scheduler_metavar,
                        type=str,
                        help="Learning rate scheduler")
    args = parser.parse_args()

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


def get_train_params(args):
    train_params = {}
    train_params["precision"] = 16 if args.fp16 else 32
    if (isinstance(args.gpus, int) and args.gpus > 1) or (isinstance(args.gpus, list) and len(args.gpus) > 1):
        train_params["distributed_backend"] = "ddp"
    else:
        train_params["distributed_backend"] = None
    train_params["accumulate_grad_batches"] = args.grad_accum
    train_params['track_grad_norm'] = -1
    train_params['limit_val_batches'] = args.limit_val_batches
    train_params['val_check_interval'] = args.val_check_interval
    train_params['gpus'] = args.gpus
    train_params['max_epochs'] = args.num_epochs
    train_params['log_every_n_steps'] = log_every_n_steps
    return train_params

# LINEに通知する関数
def line_notify(message):
    line_notify_token = 'Jou3ZkH4ajtSTaIWO3POoQvvCJQIdXFyYUaRKlZhHMI'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)


def main():
    try:
        args = parse_args()
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

        if args.test_only:
            print('loading model...')
            model = Specter.load_from_checkpoint(args.test_checkpoint)
            trainer = pl.Trainer(
                gpus=args.gpus, limit_val_batches=args.limit_val_batches)
            trainer.test(model)

        else:
            # labelList = ["title", "bg", "obj", "method", "res"]
            # IDにする
            label_dict = {
                "title": 1,
                "bg": 2,
                "obj": 3,
                "method": 4,
                "res": 5,
                "other": 6,
            }
            num_label_dict = {
                "1": "title",
                "2": "bg",
                "3": "obj",
                "4": "method",
                "5": "res",
                "6": "other"
            }
            args.label_dict = label_dict
            args.num_label_dict = num_label_dict

            model = Specter(args)

            # default logger used by trainer
            logger = TensorBoardLogger(
                save_dir=args.save_dir,
                version=args.version,
                name='pl-logs'
            )

            # second part of the path shouldn't be f-string
            dirPath = f'/workspace/dataserver/model_outputs/specter/{args.method}_{logger.version}/'
            filepath = dirPath + \
                'checkpoints/ep-{epoch}_avg_val_loss-{avg_val_loss:.3f}'
            checkpoint_callback = ModelCheckpoint(
                filepath=filepath,
                save_top_k=4,
                verbose=True,
                monitor='avg_val_loss',  # monitors metrics logged by self.log.
                mode='min',
                prefix=''
            )

            extra_train_params = get_train_params(args)
            # wandb_logger = WandbLogger(project="pretrain_label-attn-scibert")
            wandb_logger = WandbLogger(project="finetune_label-attn-scibert")
            wandb_logger.watch(model)
            trainer = pl.Trainer(logger=wandb_logger,
                                 checkpoint_callback=checkpoint_callback,
                                 **extra_train_params)

            trainer.fit(model)

            line_notify("172.21.64.47:" + os.path.basename(__file__) + "が終了")

            torch.cuda.empty_cache()

    except Exception as e:
        print(traceback.format_exc())
        message = "172.21.65.47: Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        line_notify(message)


if __name__ == '__main__':
    main()
