# transformers, pytorch
from transformers.optimization import (
    Adafactor,
)
from transformers import AdamW
import torch

# basic python packages

# inc
from inc.qkv_pooling import AttnPhi, make_pad_mask
from inc.const import label_dict, num_label_dict
from inc.SpecterOrigin import SpecterOrigin
from inc.MyDataset import PredSscDataset

class SpecterAttnAspect(SpecterOrigin):
    def __init__(self, init_args={}):
        super().__init__(init_args)
        del self.attn_pooling
        self.attn_title= AttnPhi(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)
        self.attn_bg = AttnPhi(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)
        # self.attn_obj = AttnPhi(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)
        self.attn_method = AttnPhi(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)
        self.attn_res = AttnPhi(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""        
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
        attn_params = self.make_parameter_group(self.attn_title)
        optimizer_grouped_parameters[0]['params'].extend(attn_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(attn_params['decay'])
        attn_params = self.make_parameter_group(self.attn_bg)
        optimizer_grouped_parameters[0]['params'].extend(attn_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(attn_params['decay'])
        # attn_params = self.make_parameter_group(self.attn_obj)
        # optimizer_grouped_parameters[0]['params'].extend(attn_params['no_decay'])
        # optimizer_grouped_parameters[1]['params'].extend(attn_params['decay'])
        attn_params = self.make_parameter_group(self.attn_method)
        optimizer_grouped_parameters[0]['params'].extend(attn_params['no_decay'])
        optimizer_grouped_parameters[1]['params'].extend(attn_params['decay'])
        attn_params = self.make_parameter_group(self.attn_res)
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
                    lengths = torch.tensor([label_last_hidden_state_tensor.size(0)])
                    label_last_hidden_state_padding_mask = make_pad_mask(lengths).to(device=self.hparams.device)
                    if label == 'bg':
                        label_pooling[label] = self.attn_bg(label_last_hidden_state_tensor.unsqueeze(0), label_last_hidden_state_padding_mask, self.hparams.is_key_transform)[0]
                    # elif label == 'obj':
                    #     label_pooling[label] = self.attn_obj(label_last_hidden_state_tensor.unsqueeze(0), label_last_hidden_state_padding_mask, self.hparams.is_key_transform)[0]
                    elif label == 'method':
                        label_pooling[label] = self.attn_method(label_last_hidden_state_tensor.unsqueeze(0), label_last_hidden_state_padding_mask, self.hparams.is_key_transform)[0]
                    elif label == 'res':
                        label_pooling[label] = self.attn_res(label_last_hidden_state_tensor.unsqueeze(0), label_last_hidden_state_padding_mask, self.hparams.is_key_transform)[0]
                    elif label == 'title':
                        label_pooling[label] = self.attn_title(label_last_hidden_state_tensor.unsqueeze(0), label_last_hidden_state_padding_mask, self.hparams.is_key_transform)[0]
                else:
                    label_pooling[label] = None

            batch_label_pooling.append(label_pooling)

        return batch_label_pooling