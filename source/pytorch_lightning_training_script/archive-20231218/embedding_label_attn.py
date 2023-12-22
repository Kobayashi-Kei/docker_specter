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
from finetune_attn_label import Specter
from inc.MyDataset import MyDataset
from inc.util import find_model_checkpoint, load_data, save_data

"""
SPECTER + Attntion Pooling を用いて、BERTの最終層の全ての出力を用いて
観点ごとの論文埋め込みを取得する
"""
def main():
    # 以下をモデルに合わせて変更する
    # modelType = "average_pooling"
    # modelParamPath = f"../dataserver/model_outputs/specter/20230503/version_average_pooling/checkpoints/*"
    modelType = "finetune_label-attn-scibert-correct-totalstep"
    modelParamPath = f"../dataserver/model_outputs/specter/finetune_label-attn-scibert-correct-totalstep/checkpoints" + "/*"

    # Axcellのデータサイズ(基本medium)
    size = "medium"

    # 用いる観点をリストで入力
    labelList = ["title", "bg", "obj", "method", "res"]

    # モデルパラメータのパス
    modelCheckpoint = find_model_checkpoint(modelParamPath)

    outputName = modelType

    # 入力（埋め込む論文アブストラクトデータ）
    dirPath = "../dataserver/axcell/" + size
    dataPath = dirPath + "/paperDict.json"
    labeledAbstPath = dirPath + "/labeledAbst.json"

    # 出力（文埋め込み）
    outputDirPath = dirPath + "-" + outputName + "/"
    outputEmbLabelDirPath = outputDirPath + "embLabel/"
    if not os.path.exists(outputDirPath):
        os.mkdir(outputDirPath)
    if not os.path.exists(outputEmbLabelDirPath):
        os.mkdir(outputEmbLabelDirPath)

    sscInputPath = dirPath + "/inputForSSC.jsonl"
    sscOutputPath = dirPath + "/resultSSC.txt"

    # データセットをロード
    paperDict = load_data(dataPath)
    save_data(paperDict, outputDirPath + "paperDict.json")

    # 分類されたアブストラクトをロード
    labeledAbstDict = load_data(labeledAbstPath)

    shutil.copy(labeledAbstPath, outputDirPath)

    # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
    for title in labeledAbstDict:
        labeledAbstDict[title]["title"] = title

    # SSCの結果をロード & 整形
    with open(sscInputPath, 'r') as f:
        sscInput = f.readlines()
    with open(sscOutputPath, 'r') as f:
        sscOutput = f.readlines()

    # アブスト分類の結果を読み込んで、扱いやすいように整形
    sscPaperDict = {}
    for i, line in enumerate(sscOutput):
        line = json.loads(line)
        label = line[0]
        text_list = line[1]

        inputLine = json.loads(sscInput[i])
        title = inputLine['title']

        for i, text_label_pair in enumerate(text_list):
            if text_label_pair[1] == 'background_label':
                text_list[i][1] = 'bg'
            elif text_label_pair[1] == 'method_label':
                text_list[i][1] = 'method'
            elif text_label_pair[1] == 'objective_label':
                text_list[i][1] = 'obj'
            elif text_label_pair[1] == 'result_label':
                text_list[i][1] = 'res'
            elif text_label_pair[1] == 'other_label':
                text_list[i][1] = 'other'

        sscPaperDict[title] = text_list

    try:
        # 出力用の辞書
        labeledAbstEmbedding = {}

        # モデルの初期化
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = Specter.load_from_checkpoint(modelCheckpoint)
        model.cuda()
        model.eval()

        my_dataset = MyDataset(None, None, None)

        print("--path: {}--".format(modelCheckpoint))

        # 埋め込み
        for title, paper in paperDict.items():
            labeledAbstEmbedding[title] = {}
            ssc = sscPaperDict[title]

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
            output = model.bert(**input)['last_hidden_state']
            # print(output)
            # print(model.bert(**input))

            _, label_list_for_words = my_dataset.tokenize_with_label(title_abs, ssc, tokenizer)
            label_pooling = model.label_pooling(output, torch.tensor([label_list_for_words]))[0]

            for label in labelList:
                if label_pooling[label] == None or len(label_pooling[label]) == 0:
                    labeledAbstEmbedding[title][label] = None
                    continue
                # 出力用の辞書に格納
                labeledAbstEmbedding[title][label] = label_pooling[label].tolist()

        # ファイル出力
        # labeledAbstSpecter.jsonの名称は評価プログラムでも使っているため変更しない
        with open(outputEmbLabelDirPath + "labeledAbstSpecter.json", 'w') as f:
            json.dump(labeledAbstEmbedding, f, indent=4)

        message = "【完了】shape-and-emmbedding.py"
        lineNotifier.line_notify(message)

    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)


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
    return None


if __name__ == '__main__':
    main()
