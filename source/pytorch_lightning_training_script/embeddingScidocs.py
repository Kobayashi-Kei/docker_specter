import json
import traceback
import lineNotifier
from transformers import AutoTokenizer
import os
from pretrain_lstm import Specter
import time
import shutil
import glob


def main():
    # 以下をモデルに合わせて変更する
    modelType = "lstm"
    modelParamPath = f"save/lstm-47/checkpoints/*"

    # モデルパラメータのパス
    epoch = 1
    files = glob.glob(modelParamPath)
    for filePath in files:
        if f"ep-epoch={epoch}" in filePath:
            modelCheckpoint = filePath
            break

    outputName = modelType

    # 入力（埋め込む論文アブストラクトデータ）
    dirPath = "../scidocs/scidocs/data/"
    mag_mesh = f"{dirPath}/paper_metadata_mag_mesh.json"
    recomm = f"{dirPath}/paper_metadata_recomm.json"
    view_cite_read = f"{dirPath}/paper_metadata_view_cite_read.json"

    try:
        # モデルの初期化
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = Specter.load_from_checkpoint(modelCheckpoint)
        # model.cuda(1) # 2個目のGPUを使う場合
        model.cuda()
        model.eval()
        print("--path: {}--".format(modelCheckpoint))

        for dataPath in [mag_mesh, recomm, view_cite_read]:
            # データセットをロード
            with open(dataPath, 'r') as f:
                paperDict = json.load(f)

            if dataPath == mag_mesh:
                outputPath = f"../scidocs/scidocs/data/{outputName}-embeddings/cls.jsonl"
            elif dataPath == recomm:
                outputPath = f"../scidocs/scidocs/data/{outputName}-embeddings/recomm.jsonl"
            elif dataPath == view_cite_read:
                outputPath = f"../scidocs/scidocs/data/{outputName}-embeddings/user-citation.jsonl"

            with open(outputPath, 'w') as f:
                for id, paper in paperDict.items():
                    # concatenate title and abstract
                    # abstractが無いデータがあるから、'abstract'もしくは''を使う
                    title_abs = paper['title'] + \
                        tokenizer.sep_token + (paper.get('abstract') or '')

                    # preprocess the input
                    inputs = tokenizer(title_abs, padding=True, truncation=True,
                                       return_tensors="pt", max_length=512).to('cuda:0')
                    result = model(**inputs)

                    # print(embeddings)
                    # print(embeddings.size())
                    # exit()
                    # 1行にjson形式で出力する
                    # resultはバッチで出力されるようになっているが、
                    # 今回はバッチで入力していないため0番目の要素を取り出す
                    # jsonにエンコードするためにtensor形式からリストに変換する
                    line = {"paper_id": id, "embedding": result[0].tolist()}
                    json.dump(line, f)
                    f.write('\n')

        message = "【完了】" + os.path.basename(__file__)
        lineNotifier.line_notify(message)

    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)


if __name__ == '__main__':
    main()
