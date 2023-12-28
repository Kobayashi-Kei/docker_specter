# transformers, pytorch
from transformers.optimization import (
    Adafactor,
)
from transformers import AdamW
import torch

# inc
from inc.qkv_pooling_preds import AttnPhiPreds, make_pad_mask
from inc.const import label_dict, num_label_dict
from inc.SpecterOrigin import SpecterOrigin


class SpecterAttnAspectPreds(SpecterOrigin):
    def __init__(self, init_args={}):
        super().__init__(init_args)
        del self.attn_pooling
        self.attn_title= AttnPhiPreds(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)
        self.attn_bg = AttnPhiPreds(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)
        self.attn_method = AttnPhiPreds(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)
        self.attn_res = AttnPhiPreds(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""        
        # weight decay 重み減衰：過学習を減らすためにparameterの自由度を減らす
        optimizer_grouped_parameters = [
            {
                "params": [],
                "weight_decay": 0.0,
            },
            {
                "params": [],
                "weight_decay": self.hparams.weight_decay,
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
    
    def label_pooling(self, batch_last_hidden_state, batch_position_label_preds_list):
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
            position_label_preds = batch_position_label_preds_list[b]
            lengths = last_hidden_state.size(0)
            padding_mask = make_pad_mask(torch.tensor([lengths])).to(device=self.hparams.device)
            # print(position_label_list)

            for label in label_dict:
                if label == "obj": continue
                label_preds_list = [0 for i in range(lengths)]

                # 各観点に関して各単語のPredsのリストを生成する．
                for i, preds in enumerate(position_label_preds):
                    label_preds_list[i] = preds[label_dict[label]]
                
                if label == 'bg':
                    label_pooling[label] = self.attn_bg(last_hidden_state.unsqueeze(0), torch.tensor(label_preds_list).unsqueeze(0) ,padding_mask, self.hparams.is_key_transform)[0]
                elif label == 'method':
                    label_pooling[label] = self.attn_method(last_hidden_state.unsqueeze(0), torch.tensor(label_preds_list).unsqueeze(0), padding_mask, self.hparams.is_key_transform)[0]
                elif label == 'res':
                    label_pooling[label] = self.attn_res(last_hidden_state.unsqueeze(0), torch.tensor(label_preds_list).unsqueeze(0), padding_mask, self.hparams.is_key_transform)[0]
                elif label == 'title':
                    label_pooling[label] = self.attn_title(last_hidden_state.unsqueeze(0), torch.tensor(label_preds_list).unsqueeze(0), padding_mask, self.hparams.is_key_transform)[0]

            batch_label_pooling.append(label_pooling)

        return batch_label_pooling