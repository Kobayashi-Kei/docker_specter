# python module
import json
import traceback
import os
import time
import shutil
import glob

# torch
import torch

# transformers
from transformers import AutoTokenizer, AutoModel

# my module
import lineNotifier
from pretrain_average_pooling import Specter
from inc.MyDataset import MyDataset
from inc.util import find_model_checkpoint, load_data, save_data, prepareData, save_embedding
from inc.util import labelList

"""
SPECTER + Average Pooling を用いて、BERTの最終層の全ての出力を用いて
観点ごとの論文埋め込みを取得する
"""


def main():
    from finetune_average_pooling_label import Specter

    # 以下をモデルに合わせて変更する
    # modelType = "average_pooling"
    modelType = "finetune_label-average_pooling-scibert"

    try:
        # モデルチェックポイントのパスの取得
        modelParamPath = f"../dataserver/model_outputs/specter/{modelType}/*"
        modelCheckpoint = find_model_checkpoint(modelParamPath)

        # モデルの初期化
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = Specter({})
        model.load_state_dict(torch.load(modelCheckpoint)['state_dict'])
        # model.cuda()
        model.eval()
        print("--path: {}--".format(modelCheckpoint))
        
        paperDict, sscPaperDict, outputEmbLabelDirPath = prepareData(modelType)
        labeledAbstEmbedding = embedding(model, tokenizer, paperDict, sscPaperDict)
        save_embedding(labeledAbstEmbedding, outputEmbLabelDirPath)

        lineNotifier.line_notify("【完了】shape-and-emmbedding.py")

    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)
