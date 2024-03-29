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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
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

# wandb
from pytorch_lightning.loggers import WandbLogger
import wandb

import numpy as np
import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from torchviz import make_dot
# from IPython.display import display

"""
単語位置出力を観点ごとにLSTMで集約して観点埋め込みを生成し，
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

bertOutputSize = 768

"""
自分で定義するデータセット
"""
class MyData(Dataset):
    def __init__(self, data, paper_dict, ssc_result_dict, label_dict, tokenizer, block_size=100):
        self.data_instances = data
        self.paper_dict = paper_dict
        self.ssc_result_dict = ssc_result_dict
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.label_dict = label_dict

    def __len__(self):
        return len(self.data_instances)

    def __getitem__(self, index):
        data_instance = self.data_instances[index]
        return self.retTransformersInput(data_instance, self.tokenizer)

    def retTransformersInput(self, data_instance, tokenizer):
        # print("data_instance: ", data_instance)
        source_title = re.sub(r'\s+', ' ', data_instance["source"])
        pos_title = re.sub(r'\s+', ' ', data_instance["pos"])
        neg_title = re.sub(r'\s+', ' ', data_instance["neg"])

        source_title_abs = self.paper_dict[source_title]["title"] + \
            tokenizer.sep_token + \
            self.paper_dict[source_title]["abstract"]
        sourceEncoded = self.tokenizer(
            source_title_abs,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        pos_title_abs = self.paper_dict[pos_title]["title"] + \
            tokenizer.sep_token + \
            self.paper_dict[pos_title]["abstract"]
        posEncoded = self.tokenizer(
            pos_title_abs,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        neg_title_abs = self.paper_dict[neg_title]["title"] + \
            tokenizer.sep_token + \
            self.paper_dict[neg_title]["abstract"]
        negEncoded = self.tokenizer(
            neg_title_abs,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        source_input = {
            'input_ids': sourceEncoded["input_ids"][0],
            'attention_mask': sourceEncoded["attention_mask"][0],
            'token_type_ids': sourceEncoded["token_type_ids"][0]
        }
        pos_input = {
            'input_ids': posEncoded["input_ids"][0],
            'attention_mask': posEncoded["attention_mask"][0],
            'token_type_ids': posEncoded["token_type_ids"][0]
        }
        neg_input = {
            'input_ids': negEncoded["input_ids"][0],
            'attention_mask': negEncoded["attention_mask"][0],
            'token_type_ids': negEncoded["token_type_ids"][0]
        }

        source_position_label = self.tokenize_with_label(
            source_title_abs, self.ssc_result_dict[source_title], self.tokenizer)
        pos_position_label = self.tokenize_with_label(
            pos_title_abs, self.ssc_result_dict[pos_title], self.tokenizer)
        neg_position_label = self.tokenize_with_label(
            neg_title_abs, self.ssc_result_dict[neg_title], self.tokenizer)

        ret_source = {
            "input": source_input,
            "position_label_list": torch.tensor(source_position_label[1])
        }
        ret_pos = {
            "input": pos_input,
            "position_label_list": torch.tensor(pos_position_label[1])
        }
        ret_neg = {
            "input": neg_input,
            "position_label_list": torch.tensor(neg_position_label[1])
        }


        return ret_source, ret_pos, ret_neg

    def tokenize_with_label(self, sentence, ssc_result, tokenizer):
        tokenized_input = tokenizer(
            sentence,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # input['input_ids'][0](トークナイズされた入力文)と同じ長さで、その位置のトークンのラベルを格納するリスト
        # label_list_for_words = [None for i in range(len(
        #     tokenized_input['input_ids'][0].tolist()))]

        # 最大長に合わせる
        # 【注意】 0ではなくNoneを入れていたところデータローダーの制約に引っかかってエラーとなる
        label_list_for_words = [0 for i in range(512)]

        # titleの位置にラベルを格納する
        # [SEP]の位置を特定する
        sep_pos = tokenized_input['input_ids'][0].tolist().index(102)
        for i in range(1, sep_pos):
            # label_list_for_words[i] = 'title'
            label_list_for_words[i] = 1

        # [SEP]と[CLS]は-1
        label_list_for_words[0] = -1
        label_list_for_words[sep_pos] = -1
        try:
            # 各単語の観点をlabel_positionsに格納
            for text_label_pair in ssc_result:
                text = text_label_pair[0]
                label = self.label_dict[text_label_pair[1]]

                # 1文単位でtokenizeする
                tokenizedText = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512
                )
                # tokenizeされたテキストから先頭の101([CLS])と末尾の102([SEP])を取り除く
                tokenizedText_input_ids = tokenizedText['input_ids'][0][1:-1].tolist()

                start, end = self.find_subarray(
                    tokenized_input['input_ids'][0].tolist(), tokenizedText_input_ids)
                # たまに見つからないときがある（なぜ？）
                if start and end:
                    for i in range(start, end+1):
                        label_list_for_words[i] = label

            return tokenized_input, label_list_for_words

        except Exception as e:
            print("text: ", text)
            print("tokenized_input['input_ids'][0].tolist()",
                  tokenized_input['input_ids'][0].tolist())
            print("tokenized_input_ids", tokenizedText_input_ids)
            print(traceback.format_exc())

    def find_subarray(self, arr, subarr):
        n = len(arr)
        m = len(subarr)

        # サブ配列が配列に含まれるかどうかを調べる
        for i in range(n - m + 1):
            j = 0
            while j < m and arr[i + j] == subarr[j]:
                j += 1
            if j == m:
                return (i, i + m - 1)

        # サブ配列が配列に含まれない場合は None を返す
        return (None, None)


"""
ロス計算を行うモジュール
"""
class TripletLoss(nn.Module):
    """
    Triplet loss: copied from  https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/model.py#L159 without any change
    """

    def __init__(self, margin=1.0, distance='l2-norm', reduction='mean'):
        """
        Args:
            margin: margin (float, optional): Default: `1`.
            distance: can be `l2-norm` or `cosine`, or `dot`
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction

    def forward(self, query, positive, negative):
        if self.distance == 'l2-norm':
            distance_positive = F.pairwise_distance(query, positive)
            distance_negative = F.pairwise_distance(query, negative)
            losses = F.relu(distance_positive -
                            distance_negative + self.margin)
        elif self.distance == 'cosine':  # independent of length
            distance_positive = F.cosine_similarity(query, positive)
            distance_negative = F.cosine_similarity(query, negative)
            losses = F.relu(-distance_positive +
                            distance_negative + self.margin)
        elif self.distance == 'dot':  # takes into account the length of vectors
            shapes = query.shape
            # batch dot product
            distance_positive = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                positive.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0], )
            distance_negative = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                negative.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0], )
            losses = F.relu(-distance_positive +
                            distance_negative + self.margin)
        else:
            raise TypeError(
                f"Unrecognized option for `distance`:{self.distance}")
        #
        # debug print
        #
        # print("query: ", query)
        # print("positive: ", positive)
        # print("distance_positive: ", distance_positive)
        # print("distance_negative: ", distance_negative)
        print("losses: ", losses)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise TypeError(
                f"Unrecognized option for `reduction`:{self.reduction}")


"""
モデルのクラス
"""
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
        # self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        
        # SPECTERを初期値とする場合
        self.bert = AutoModel.from_pretrained("allenai/specter")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

        self.tokenizer.model_max_length = self.bert.config.max_position_embeddings
        self.hparams.seqlen = self.bert.config.max_position_embeddings
        self.triple_loss = TripletLoss(margin=float(init_args.margin))
        
        # BERTの出力トークンを統合するレイヤー
        # input_size: 各時刻における入力ベクトルのサイズ、ここではBERTの出力の768次元になる
        # hidden_size: メモリセルとかゲートの隠れ層の次元、出力のベクトルの次元もこの値になる（Batch_size, sequence_length, hidden_size)
        #   chatGPTによると一般的には、LSTMの隠れ層の次元は、入力データの次元と同じであることが多い
        self.lstm = nn.LSTM(input_size=bertOutputSize,
                            hidden_size=bertOutputSize, batch_first=True)


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
        path = "/workspace/dataserver/axcell/large/specter/paper/triple-" + split + ".json"
        with open(path, 'r') as f:
            data = json.load(f)

        path = "/workspace/dataserver/axcell/large/paperDict.json"
        with open(path, 'r') as f:
            paper_dict = json.load(f)

        path = "/workspace/dataserver/axcell/large/result_ssc.json"
        with open(path, 'r') as f:
            ssc_result_dict = json.load(f)

        dataset = MyData(data, paper_dict, ssc_result_dict,
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
        model_parameters = list(self.bert.named_parameters()) + \
            list(self.lstm.named_parameters())
        
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
        # source_embedding = self.model(**batch[0])[1] # [1]はpooler_outputのこと　https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
        # pos_embedding = self.model(**batch[1])[1]
        # neg_embedding = self.model(**batch[2])[1]
        source_embedding = self.bert(**batch[0]["input"])['last_hidden_state']
        pos_embedding = self.bert(**batch[1]["input"])['last_hidden_state']
        neg_embedding = self.bert(**batch[2]["input"])['last_hidden_state']
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
        for b in range(len(source_label_pooling)): # batchsizeの数だけループ
            label_loss = 0
            valid_label_list = []
            for label in self.hparams.label_dict:
                if not source_label_pooling[b][label] == None and not pos_label_pooling[b][label] == None and not neg_label_pooling[b][label] == None:
                    valid_label_list.append(label)
                    # debug print
                    print("--" , label)
                    label_loss += self.triple_loss(source_label_pooling[b][label], pos_label_pooling[b][label], neg_label_pooling[b][label])

            if len(valid_label_list) > 0:
                batch_label_loss += label_loss/len(valid_label_list)
                label_loss_calculated_count += 1
        
        """
        一致する観点がなければlossは0
        """
        if label_loss_calculated_count > 0:
            loss = batch_label_loss / label_loss_calculated_count
        else:
            loss = 0
        
        if self.debug:
            return {"loss": loss}

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        self.log('train_loss', loss, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)
        self.log('rate', lr_scheduler.get_last_lr()
                 [-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # source_embedding = self.model(**batch[0])[1] # [1]はpooler_outputのこと　https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
        # pos_embedding = self.model(**batch[1])[1]
        # neg_embedding = self.model(**batch[2])[1]
        source_embedding = self.bert(**batch[0]["input"])['last_hidden_state']
        pos_embedding = self.bert(**batch[1]["input"])['last_hidden_state']
        neg_embedding = self.bert(**batch[2]["input"])['last_hidden_state']
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

                    print(source_label_pooling[b][label])
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
            loss = 0

        if self.debug:
            return {"loss": loss}

        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return {'val_loss': loss}

    def _eval_end(self, outputs) -> tuple:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
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
                    # outputに各単語に対応する隠れ層の出力（単語数×次元数のtensor）, _ に最後の単語に対応する隠れ層の出力とCtのタプル
                    # つまり，outputの最後の要素を予測に利用する
                    # 参考: https://qiita.com/m__k/items/841950a57a0d7ff05506#%E3%83%A2%E3%83%87%E3%83%AB%E5%AE%9A%E7%BE%A9
                    output, _ = self.lstm(
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
            wandb.init(project='SPECTER-LSTM-label')
            wandb_logger = WandbLogger(project="SPECTER-LSTM-share-label",
                                       tags=["SPECTER", "LSTM", "label"])

            trainer = pl.Trainer(logger=wandb_logger,
                                 checkpoint_callback=checkpoint_callback,
                                 **extra_train_params)

            # 計算グラフの可視化
            wandb.watch(model, log='all')

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
