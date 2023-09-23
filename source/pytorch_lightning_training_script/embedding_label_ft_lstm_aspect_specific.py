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
from transformers import AutoTokenizer

# my module
import lineNotifier
from finetune_lstm_aspect_specific_label import Specter

"""
SPECTER + LSTM を用いて、BERTの最終層の全ての出力を用いて
観点ごとの論文埋め込みを取得する
"""


def main():
    # 以下をモデルに合わせて変更する
    modelType = "specter_lstm"
    modelParamPath = f"../dataserver/model_outputs/specter/paper_specter_lstm/checkpoints" + "/*"

    # Axcellのデータサイズ(基本medium)
    size = "medium"

    # 用いる観点をリストで入力
    labelList = ["title", "bg", "obj", "method", "res"]

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
    with open(dataPath, 'r') as f:
        paperDict = json.load(f)

    with open(outputDirPath + "paperDict.json", "w") as f:
        json.dump(paperDict, f, indent=4)

    # 分類されたアブストラクトをロード
    with open(labeledAbstPath, 'r') as f:
        labeledAbstDict = json.load(f)

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

        # if "No Need" in title:
        #     print(title)
    # print(sscPaperDict)
    # exit()

    try:
        # 出力用の辞書
        labeledAbstEmbedding = {}

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

            # input['input_ids'][0](トークナイズされた入力文)と同じ長さで、その位置のトークンのラベルを格納するリスト
            label_positions = [None for i in range(len(
                input['input_ids'][0].tolist()))]

            # titleの位置にラベルを格納する
            # SEPトークンの位置を特定する
            sep_pos = input['input_ids'][0].tolist().index(102)
            for i in range(1, sep_pos):
                label_positions[i] = 'title'
            # print(input['input_ids'][0].tolist())
            # print(sep_pos)
            # print(label_positions)
            # exit()

            # 各トークンの観点をlabel_positionsに格納
            for text_label_pair in ssc:
                text = text_label_pair[0]
                label = text_label_pair[1]

                # 1文単位でtokenizeする
                tokenizedText = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512
                )
                # 先頭の101([CLS])と末尾の102([SEP])を取り除く
                tokenizedText_input_ids = tokenizedText['input_ids'][0][1:-1].tolist()
                # print(tokenizedText_input_ids)
                # exit()

                start, end = find_subarray(
                    input['input_ids'][0].tolist(), tokenizedText_input_ids)
                for i in range(start, end+1):
                    label_positions[i] = label
                # print(start, end)
                # print(label_positions)
            # print(input)

            # exit()
            # 各トークンをBERTに通す
            input = input.to('cuda:0')
            out = model.forward(**input)[0][0]

            # print(output)
            # print(output.size())
            # exit()

            # output_format = output.tolist()
            # print(output)
            # print(count, labeledAbstDict[title][label])

            # 観点ごとにBERT最終層出力を配列にまとめる
            label_last_hideen_state = {label: [] for label in labelList}
            for i, tensor in enumerate(out):
                # print(tensor)
                # [CLS]or [SEP]の時
                if label_positions[i] == None or label_positions[i] == "other":
                    continue
                label_last_hideen_state[label_positions[i]].append(
                    tensor.tolist())

            # 観点ごとのBERT出力をLSTMに通す
            for label in labelList:
                if len(label_last_hideen_state[label]) == 0:
                    labeledAbstEmbedding[title][label] = None
                    continue
                # print(label_last_hideen_state[label])
                lstmInput = torch.tensor(
                    label_last_hideen_state[label]).unsqueeze(0).to('cuda:0')
                
                if label == 'bg':
                    out, _ = model.lstm_bg(
                        lstmInput, None)
                elif label == 'obj':
                    out, _ = model.lstm_obj(
                        lstmInput, None)
                elif label == 'method':
                    out, _ = model.lstm_title(
                        lstmInput, None)
                elif label == 'res':
                    out, _ = model.lstm_res(
                        lstmInput, None)
                elif label == 'title':
                    out, _ = model.lstm_title(
                        lstmInput, None)
                # print(out[:, -1, :])
                # print(out[:, -1, :].size())
                # 出力用の辞書に格納
                labeledAbstEmbedding[title][label] = out[:, -1, :].tolist()[0]

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
