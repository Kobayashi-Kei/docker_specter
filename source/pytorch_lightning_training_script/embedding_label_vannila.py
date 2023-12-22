# python module
import traceback
import os

# torch, transformers
import torch
from transformers import AutoTokenizer

# my module
import lineNotifier

from inc.MyDataset import MyDataset
from inc.util import find_model_checkpoint, load_data, save_data, prepareData, save_embedding
from inc.util import labelList
from inc.util import embedding

"""
SPECTER + Attntion Pooling を用いて、BERTの最終層の全ての出力を用いて
観点ごとの論文埋め込みを取得する
"""
def main():
    # 以下をモデルに合わせて変更する
    # modelType = "average_pooling"
    from finetune_attn_label import Specter
    modelType = "finetune_label-attn-scibert-vannila-1ep"

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



if __name__ == '__main__':
    main()
