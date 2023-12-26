import torch
import os 
import sys
import json, codecs
import numpy as np

import torch.nn.functional as F
import wandb

from inc.const import label_dict, num_label_dict, csf_label_dict, csf_num_label_dict, csf_to_label
import inc.csf_ranking_eval as csf_rank

def tokenize(title_abst, tokenizer):
    tokenized_text = tokenizer(
                title_abst,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
    return tokenized_text

def make_cls_title_sep_abst(paper, tokenizer):
    title_abs = paper['title'] + tokenizer.sep_token + ' '.join(paper['abstract'])
    return title_abs

def list_None_to_int(l):
    for i, item in enumerate(l):
        if item == None:
            l[i] = -1
    return l

def get_label_positions(paper, tokenizer):
    # print(paper)
    tokenized_input = tokenize(make_cls_title_sep_abst(paper, tokenizer), tokenizer)
    # print(tokenized_input)
    label_positions = \
        [None for i in range(tokenized_input['input_ids'][0].size(0))]
    
    sep_pos = tokenized_input['input_ids'][0].tolist().index(102)
    for i in range(1, sep_pos):
        label_positions[i] = csf_label_dict["title"]

    # [SEP]と[CLS]は-1
    label_positions[0] = -1
    label_positions[sep_pos] = -1
    # print(paper)
    # print(tokenized_input['input_ids'][0].size(0))
    # 各トークンの観点をlabel_positionsに格納
    for i, sentence in enumerate(paper['abstract']):
        label = paper['pred_labels'][i]

        # 1文単位でtokenizeする
        tokenized_sentence = tokenizer(
            sentence,
            return_tensors="pt",
            max_length=512
        )
        # 先頭の101([CLS])と末尾の102([SEP])を取り除く
        tokenized_text_input_ids = tokenized_sentence['input_ids'][0][1:-1].tolist()
        start, end = find_subarray(
            tokenized_input['input_ids'][0].tolist(), tokenized_text_input_ids)
        # トークン数が512を超える場合はsubarrayが見つからない，breakしてそれ以降の情報は捨てる
        if start == None and end == None:
            break

        for i in range(start, end+1):
            label_positions[i] = csf_label_dict[label]

    return list_None_to_int(label_positions)

def find_subarray(arr, subarr):
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


def gen_label_pooled_output(paper, model, tokenizer, device):
    label_positions = get_label_positions(paper, tokenizer)
    tokenized_input = tokenize(make_cls_title_sep_abst(paper, tokenizer), tokenizer).to(device)
    label_pooling = model.forward(tokenized_input, torch.tensor(label_positions).unsqueeze(0))[0]
    
    return label_pooling


def calc_dist(label, query, candidate, model, tokenizer, device):
    # query
    query_label_pooled = gen_label_pooled_output(query, model, tokenizer, device)
    candidate_label_pooled = gen_label_pooled_output(candidate, model, tokenizer, device)
    # cosine_similarityは Expected 2D array のため

    if query_label_pooled[csf_to_label[label]] != None and candidate_label_pooled[csf_to_label[label]] != None:
        dist = F.cosine_similarity(query_label_pooled[csf_to_label[label]], candidate_label_pooled[csf_to_label[label]], dim=0)
    else:
        # アブストがBERTの最大長を超え，かつlabelが最大長の中に存在しない場合
        # もしくはそもそもそのラベルが存在しない場合
        dist = 1000000
    
    # print(dist)
    return dist

def eval_log_CSFCube(model, tokenizer, device, model_name):
    aggmetrics = eval_CSFCube(model, tokenizer, device, model_name)
    log_dict = {
        'CSFCube_r_precision': aggmetrics['r_precision'], 
        'CSFCube_Precision@20': aggmetrics['precision'],
        'CSFCube_recall@20': aggmetrics['recall'],
        'CSFCube_ndcg': aggmetrics['ndcg'],
        'CSFCube_ndcg@20': aggmetrics['ndcg@20'],
        'CSFCube_ndcg%20': aggmetrics['ndcg%20'],
        'CSFCube_mrr': aggmetrics['mean_reciprocal_rank'],
        'CSFCube_map': aggmetrics['mean_av_precision'],
    }
    wandb.log(log_dict)
    return log_dict

def eval_CSFCube(model, tokenizer, device, model_name):
    original_directory = os.getcwd()
    os.chdir("../CSFCube")
    model.eval()

    directory_path = f'./eval_scripts/ranked_result/{model_name}'

    # ディレクトリが存在しない場合にのみ作成します
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    # データの読み込み
    label_list = ['background', 'method', 'result']
    for label in label_list:
        pid2abstract = {}
        with codecs.open('abstracts-csfcube-preds.jsonl', 'r', 'utf-8') as absfile:
            for line in absfile:
                injson = json.loads(line.strip())
                pid2abstract[injson['paper_id']] = injson

        qpid2pool = {}
        with codecs.open(f'test-pid2anns-csfcube-{label}.json', 'r', 'utf-8') as fp:
            qpid2pool = json.load(fp)

        # ランキングの生成
        qpid2pool_ranked = {}
        for qpid, pool in qpid2pool.items():
            cand_pids = pool['cands']
            query_cand_distance = []
            for cpid in cand_pids:
                # print(pid2abstract[qpid], pid2abstract[cpid])
                dist = calc_dist(label, pid2abstract[qpid], pid2abstract[cpid], model, tokenizer, device)
                # print(dist)
                query_cand_distance.append([cpid, float(dist)])
            ranked_pool = list(sorted(query_cand_distance, key=lambda cd: cd[1]))
            qpid2pool_ranked[qpid] = ranked_pool

        # ランキング結果の出力
        with codecs.open(f'{directory_path}/test-pid2pool-csfcube-{model_name}-{label}-ranked.json', 'w', 'utf-8') as fp:
            json.dump(qpid2pool_ranked, fp)

    aggmetrics = csf_rank.csf_eval(model_name, f"eval_scripts/ranked_result/{model_name}", './', 'all')

    os.chdir(original_directory)

    return aggmetrics