# python module
import json
import traceback
import os
from inc.util import line_notify
from inc.openai import get_embedding

def main():
    # Axcellのデータサイズ(基本medium)
    size = "medium"
    outputName = "openai"

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

        # 埋め込み
        count = 0
        for title, paper in paperDict.items():

            # Title + Abstract
            title_abs = paper['title'] + (paper.get('abstract') or '')
            
            outputTitleAbstEmbedding[title] = get_embedding(title_abs)

            count += 1
            print(str(count))
        # ファイル出力
        # labeledAbstSpecter.jsonの名称は評価プログラムでも使っているため変更しない
        with open(outputEmbeddingDirPath + "titleAbstSpecter.json", 'w') as f:
            json.dump(outputTitleAbstEmbedding, f, indent=4)

        message = "【完了】" + os.path.basename(__file__)
        line_notify(message)

    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        line_notify(message)


if __name__ == '__main__':
    main()


