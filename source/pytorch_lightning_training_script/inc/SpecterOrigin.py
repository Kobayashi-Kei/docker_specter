# transformers, pytorch
from transformers.optimization import (
    Adafactor,
)
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW

from torch.utils.data import DataLoader
import torch

# basic python packages
import json
from argparse import Namespace

# inc
from inc.MyDataset import MyDataset
from inc.TripletLoss import TripletLoss
from inc.qkv_pooling import AttnPhi, make_pad_mask
from inc.const import label_dict, num_label_dict
from inc.const import arg_to_scheduler, arg_to_scheduler_choices, arg_to_scheduler_metavar
from inc.const import tokenizer_name


class SpecterOrigin(torch.nn.Module):
    def __init__(self, init_args={}):
        super().__init__()
        if isinstance(init_args, dict):
            # for loading the checkpoint, pl passes a dict (hparams are saved as dict)
            init_args = Namespace(**init_args)

        if hasattr(init_args, 'checkpoint_path'):
            checkpoint_path = init_args.checkpoint_path

        self.hparams = init_args

        # SciBERTを初期値
        self.bert = AutoModel.from_pretrained(tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # SPECTERを初期値とする場合
        # self.bert = AutoModel.from_pretrained("allenai/specter")
        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

        # Average Pooling Specterを初期値
        # modelCheckpoint = '/workspace/dataserver/model_outputs/specter/pretrain_average_pooling/checkpoints/ep-epoch=1_avg_val_loss-avg_val_loss=0.260-v0.ckpt'
        # self.bert = Specter.load_from_checkpoint(modelCheckpoint).bert
        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

        self.tokenizer.model_max_length = self.bert.config.max_position_embeddings
        self.hparams.seqlen = self.bert.config.max_position_embeddings
        if hasattr(init_args, 'margin'):
            self.triple_loss = TripletLoss(margin=float(init_args.margin))

        if hasattr(init_args, 'is_key_transform'):
            self.attn_pooling = AttnPhi(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)
        else:
            self.attn_pooling = AttnPhi(self.bert.config.hidden_size, is_key_transform=False, device=self.hparams.device)
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

    def forward(self, input, position_label_list):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.bert(
            input_ids=input['input_ids'], token_type_ids=input['token_type_ids'], attention_mask=input['attention_mask'])
        
        label_pooling = self.label_pooling(embedding['last_hidden_state'], position_label_list)

        return label_pooling

    """
    このメソッドでallennlp用のデータをロードして、トークナイズまで行う（tokentype_id, attention_maskなど)
    -> つまりこのメソッドに関わる箇所を書き換えればいい。
    """
    def _get_loader(self, split, data_name='axcell'):
        # Axcell データ
        if data_name == 'axcell':
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
        elif data_name == 'specter':
            path = "/workspace/dataserver/specterData/label/" + split + "/specter_triple-" + split + ".json"
            with open(path, 'r') as f:
                data = json.load(f)

            path = "/workspace/dataserver/specterData/label/" + split + "/specter_paperDict.json"
            with open(path, 'r') as f:
                paper_dict = json.load(f)

            path = "/workspace/dataserver/specterData/label/" + split + "/result_ssc.json"
            with open(path, 'r') as f:
                ssc_result_dict = json.load(f)

        dataset = MyDataset(data, paper_dict, ssc_result_dict,
                        label_dict, self.tokenizer)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                            shuffle=False, pin_memory=False)

        # print(loader)
        return loader

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(
            1, self.hparams.total_gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.batch_size * \
            self.hparams.grad_accum * num_devices
        # dataset_size = len(self.train_loader.dataset)
        """The size of the training data need to be coded with more accurate number"""
        dataset_size = len(self._get_loader("train", self.hparams.data_name))
        # return (dataset_size / effective_batch_size) * self.hparams.num_epochs
        return (dataset_size / effective_batch_size) * self.hparams.num_epochs*2

    def get_lr_scheduler(self):
        # default: self.hparams.lr_scheduler = liner
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        # warmup_stepsはdefaultで0. ただし，一般的には最適値の飛び越え防止や，過学習抑制のために設定した方がいいらしい
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )

        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        # old
        # no_decay = ["bias", "LayerNorm.weight"]
        # model_parameters = self.named_parameters()
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model_parameters if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        #     {
        #         "params": [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.hparams.weight_decay,
        #         # "weight_decay": 0.0,
        #     },
        # ]
        
        # new!
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
        attn_params = self.make_parameter_group(self.attn_pooling)
        optimizer_grouped_parameters[0]['params'].extend(attn_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(attn_params['decay'])

        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon
            )

        self.opt = optimizer
        
        # get_lr_scheduler内でself.optを渡す
        scheduler = self.get_lr_scheduler()

        return optimizer, scheduler

    def make_parameter_group(self, model):
        no_decay = ["bias", "LayerNorm.weight"]
        no_decay_params = []
        decay_params = []
        for n, p in model.named_parameters():
            if any(nd in n for nd in no_decay):
                # print(n)
                no_decay_params.append(p)
            else:
                # print(n)
                decay_params.append(p)

        return {'no_decay': no_decay_params, 'decay' : decay_params}

    def training_step(self, batch, batch_idx):
        # print(batch)
        source_label_pooling = self.forward(batch[0]["input"], batch[0]["position_label_list"])
        pos_label_pooling = self.forward(batch[1]["input"], batch[1]["position_label_list"])
        neg_label_pooling = self.forward(batch[2]["input"], batch[2]["position_label_list"])

        loss = self.calc_label_total_loss(source_label_pooling, pos_label_pooling, neg_label_pooling)

        # self.log('train_loss', loss, on_step=True,
        #          on_epoch=False, prog_bar=True, logger=True)
        # self.log('rate', lr_scheduler.get_last_lr()
        #          [-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        source_label_pooling = self.forward(batch[0]["input"], batch[0]["position_label_list"])
        pos_label_pooling = self.forward(batch[1]["input"], batch[1]["position_label_list"])
        neg_label_pooling = self.forward(batch[2]["input"], batch[2]["position_label_list"])

        
        loss = self.calc_label_total_loss(source_label_pooling, pos_label_pooling, neg_label_pooling)

        # wandb.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def label_pooling(self, batch_last_hidden_state, batch_position_label_list):
        batch_size = batch_last_hidden_state.size(0)
        batch_label_pooling = []

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
                    label_last_hidden_state_padding_mask = make_pad_mask(lengths).to(device=self.hparams.device)
                    # print(label_last_hidden_state_tensor.unsqueeze(0).size())
                    # print(label_last_hidden_state_padding_mask)
                    label_pooling[label] = self.attn_pooling(label_last_hidden_state_tensor.unsqueeze(0), label_last_hidden_state_padding_mask, self.hparams.is_key_transform)[0]
                    # print(label_pooling[label].size())
                else:
                    label_pooling[label] = None

            batch_label_pooling.append(label_pooling)

        return batch_label_pooling

    def calc_label_total_loss(self, source_label_pooling, pos_label_pooling, neg_label_pooling):
        pass
