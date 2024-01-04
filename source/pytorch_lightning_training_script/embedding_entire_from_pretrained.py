import json
from transformers import AutoTokenizer, AutoModel
import os



def main():
    """
    データセットのパスなどを代入
    """

    size = "medium-scincl"

    # 入力（埋め込む論文アブストラクトデータ）
    dirPath = "../dataserver/axcell/" + size
    dataPath = dirPath + "/paperDict.json"
    labeledAbstPath = dirPath + "/labeledAbst_bgobj.json"

    # 出力（文埋め込み）
    outputEmbPath = dirPath + "/embedding/"

    """
    データのロード・整形
    """
    # データセットをロード
    with open(dataPath, 'r') as f:
        paperDict = json.load(f)

    # 分類されたアブストラクトをロード
    with open(labeledAbstPath, 'r') as f:
        labeledAbstDict = json.load(f)


    # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
    for title in labeledAbstDict:
        labeledAbstDict[title]["title"] = title

    """
    観点毎のデータで学習した観点毎のSPECTERモデルで埋め込み
    """
    # 出力用
    embeddings = {}

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
    model = AutoModel.from_pretrained('malteos/scincl')
    model.cuda()
    model.eval()

    count = 0
    # 埋め込み
    
    for title, paper in paperDict.items():
        title_abs = paper['title'] + (paper.get('abstract') or '')
        input = tokenizer(
            # 文の間には[SEP]を挿入しない（挿入した方が良かったりする？）
            title_abs,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=512
            # ).to('cuda:1')
        ).to('cuda:0')

        count += 1

        output = model(**input).last_hidden_state[:, 0, :]

        outputFormat = output[0].tolist()
        # print(output.size())
        # print(output)
        # exit()
        # print(count, labeledAbstDict[title][label])
        embeddings[title] = outputFormat


    # ファイル出力
    with open(outputEmbPath + "embedding_entire.json", 'w') as f:
        json.dump(embeddings, f, indent=4)


if __name__ == '__main__':
    main()
