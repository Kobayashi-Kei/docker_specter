# python module
import json
import traceback
import os
import shutil
import glob

# torch
import torch

# transformers
from transformers import AutoTokenizer

# my module
import lineNotifier
from pretrain_lstm import Specter

"""
SPECTER + LSTM を用いて、BERTの最終層の全ての出力を用いて
タイトル + アブスト全体の埋め込みを取得する
"""


def main():
    # 以下をモデルに合わせて変更する
    modelType = "pretrain_lstm_entire"
    modelParamPath = f"../dataserver/model_outputs/specter/pretrain_lstm/checkpoints" + "/*"

    # Axcellのデータサイズ(基本medium)
    size = "medium"

    # モデルパラメータのパス
    epoch = 1
    files = glob.glob(modelParamPath)
    for filePath in files:
        if f"ep-epoch={epoch}" in filePath:
            modelCheckpoint = filePath
            break

    outputName = modelType

    # 入力（埋め込む論文アブストラクトデータ）
    dirPath = "../dataserver/axcell/" + size
    dataPath = dirPath + "/paperDict.json"

    # 出力（文埋め込み）
    outputDirPath = dirPath + "-" + outputName + "/"
    outputEmbeddingDirPath = outputDirPath + "embedding/"
    if not os.path.exists(outputDirPath):
        os.mkdir(outputDirPath)
    if not os.path.exists(outputEmbeddingDirPath):
        os.mkdir(outputEmbeddingDirPath)

    # データセットをロード
    with open(dataPath, 'r') as f:
        paperDict = json.load(f)

    with open(outputDirPath + "paperDict.json", "w") as f:
        json.dump(paperDict, f, indent=4)

    try:
        # 出力用の辞書
        outputTitleAbstEmbedding = {}

        # モデルの初期化
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = Specter.load_from_checkpoint(modelCheckpoint)
        # print(model.lstm)
        # exit()
        # model.cuda(1)
        model.cuda()
        model.eval()

        print("--path: {}--".format(modelCheckpoint))

        # 埋め込み
        for title, paper in paperDict.items():

            # Title + [SEP] + Abstract
            title_abs = paper['title'] + \
                tokenizer.sep_token + (paper.get('abstract') or '')
            input = tokenizer(
                title_abs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            # 各トークンをBERTに通す
            input = input.to('cuda:0')
            output = model.bert(**input)
            # print(output['last_hidden_state'].size())
            # >> torch.Size([1, 154, 768])

            out, _ = model.lstm(output['last_hidden_state'], None)
            # print(out.size())
            # >> torch.Size([1, 154, 768])
            # print(out[:, -1, :].size())
            # >> torch.Size([1, 768])

            outputTitleAbstEmbedding[title] = out[:, -1, :][0].tolist()

        # ファイル出力
        # labeledAbstSpecter.jsonの名称は評価プログラムでも使っているため変更しない
        with open(outputEmbeddingDirPath + "titleAbstSpecter.json", 'w') as f:
            json.dump(outputTitleAbstEmbedding, f, indent=4)

        message = "【完了】" + os.path.basename(__file__)
        lineNotifier.line_notify(message)

    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)


if __name__ == '__main__':
    main()
