import json
import traceback
from transformers import AutoTokenizer
import os
import shutil
import glob
from inc.util import line_notify
from inc.openai import get_embedding
from inc.util import labelList

def main():
     # Axcellのデータサイズ(基本medium)
    size = "medium"
    outputName = "openai"

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

    """
    データのロード・整形
    """
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

    try:
        """
        観点毎のデータで学習した観点毎のSPECTERモデルで埋め込み
        """
        # 出力用
        labeledAbstEmbedding = {}
        count = 0
        for title, paper in paperDict.items():
            labeledAbstEmbedding[title] = {}
            for i, label in enumerate(labelList):        
                if labeledAbstDict[title][label]:
                    labeledAbstEmbedding[title][label] = \
                        get_embedding(labeledAbstDict[title][label])
                    print(len(labeledAbstEmbedding[title][label]))
                    if len(labeledAbstEmbedding[title][label]) != 1536:
                        line_notify(f"{labeledAbstEmbedding[title]}: {label} is not 1536")
                else:
                    labeledAbstEmbedding[title][label] = None
            count += 1
            print(count)
            if count!=0 and count % 100==0:
                line_notify(str(count))
                
        # ファイル出力
        with open(outputEmbLabelDirPath + "labeledAbstSpecter.json", 'w') as f:
            json.dump(labeledAbstEmbedding, f, indent=4)

        message = "【完了】shape-and-emmbedding.py"
        line_notify(message)

    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        line_notify(message)


if __name__ == '__main__':
    main()
