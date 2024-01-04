# transformers, pytorch
import torch
from transformers.optimization import (
    Adafactor,
)
from transformers import AdamW
from torch.utils.data import DataLoader

# basic python 
import json
import csv

# inc
from inc.qkv_pooling_preds import AttnPhiPreds, make_pad_mask
from inc.const import label_dict, num_label_dict
from inc.SpecterOrigin import SpecterOrigin
from inc.MyDataset import PredSscDataset

class SpecterAttnPreds(SpecterOrigin):
    def __init__(self, init_args={}):
        super().__init__(init_args)
        self.attn_pooling = AttnPhiPreds(self.bert.config.hidden_size, is_key_transform=self.hparams.is_key_transform, device=self.hparams.device)

    def _get_loader(self, split, data_name='axcell'):
        # Axcell データ
        if data_name == 'axcell':
            path = "/workspace/dataserver/axcell/large/specter/paper/triple-" + split + ".json"
            with open(path, 'r') as f:
                data = json.load(f)

            path = "/workspace/dataserver/axcell/large/paperDict.json"
            with open(path, 'r') as f:
                paper_dict = json.load(f)

            path = "/workspace/dataserver/axcell/large/ssc_result_label_preds.jsonl"
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

            path = "/workspace/dataserver/specterData/label/" + split + "/result_ssc_preds.json"
            with open(path, 'r') as f:
                ssc_result_dict = json.load(f)

        # SciNCL データ
        elif data_name == 'scincl':
            path = "/workspace/dataserver/scincl/" + split + "_metadata.jsonl"
            paper_dict = {}
            paper_id_to_title = {}
            with open(path, 'r') as f:
                for line in f:
                    paper_jsonl = json.loads(line)
                    if paper_jsonl['abstract'] == None:
                        paper_jsonl['abstract'] = ''
                    paper_dict[paper_jsonl["title"]] = paper_jsonl
                    paper_id_to_title[paper_jsonl['paper_id']] = paper_jsonl["title"]

            data = []
            path = "/workspace/dataserver/scincl/" + split + "_triples.csv"
            with open(path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=['source', 'pos', 'neg'])
                next(reader)  # 最初の行（ヘッダー）をスキップする
                for row in reader:
                    data.append({
                        "source": paper_id_to_title[row['source']],
                        "pos": paper_id_to_title[row['pos']],
                        "neg": paper_id_to_title[row['neg']]
                    })

            path = "/workspace/dataserver/scincl/result_ssc_preds.json"
            with open(path, 'r') as f:
                ssc_result_dict = json.load(f)

        dataset = PredSscDataset(data, paper_dict, ssc_result_dict,
                        label_dict, self.tokenizer)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                            shuffle=False, pin_memory=False)

        # print(loader)
        return loader
    
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
            # print(last_hidden_state)
            # print('----------------------------------')
            # print(position_label_preds)
            
            for label in label_dict:
                if label == "obj": continue

                # その観点に分類される確率（重み）を全単語に関して格納する
                label_preds_list = [0 for i in range(lengths)]

                # 各観点に関して各単語のPredsのリストを生成する．
                for i, preds in enumerate(position_label_preds):
                    label_preds_list[i] = preds[label_dict[label]]
                
                label_pooling[label] = self.attn_pooling(last_hidden_state.unsqueeze(0), torch.tensor(label_preds_list).unsqueeze(0) ,padding_mask, self.hparams.is_key_transform)[0]

            batch_label_pooling.append(label_pooling)

        return batch_label_pooling