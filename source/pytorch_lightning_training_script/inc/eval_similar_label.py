import json

import wandb

from inc.recom import allPaperDataClass, testPaperDataClass
from sklearn.metrics.pairwise import cosine_similarity


labelList = ['bg', 'method', 'res']


def main():
    # embedding_dir = "medium-finetune_specter_label-attn_aspect_specific_key_transform_margin1-lr2e-6-18900"
    embedding_dir = "medium"
    embedding_dir = "medium-openai"
    # embedding_dir = "medium-finetune_specter_label-attn_aspect_specific_average__key_transform_margin1-lr2e-6-200"
    embedding_dir = "medium-pretrain_label-attn_aspect_specific_average_key_transform_margin2-lr2e-6-340000"

    eval_similar_label(embedding_dir)

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def eval_log_similar_label(embedding_dir, dirPath):
    accuracy_match, accuracy_inclusion = eval_similar_label(embedding_dir, dirPath)
    wandb.log({"Similar Label Acc Match": accuracy_match})
    wandb.log({"Similar Label Acc Inclusion": accuracy_inclusion})
    return accuracy_match, accuracy_inclusion

def eval_similar_label(embedding_dir, dirPath):
    """
    事前にembedding()しておく必要あり
    """
    embedding_path = dirPath + embedding_dir

    allPaperData = allPaperDataClass()
    testPaperData = testPaperDataClass()
    allPaperData.paperDict = load_data(dirPath + "medium/paperDict_label_ano.json")
    labeledAbstDict = load_data(embedding_path + "/embLabel/labeledAbstSpecter.json")

    """
    データの整形
    """
    for title, paper in allPaperData.paperDict.items():
        # Vectorizerに合うようにアブストラクトのみをリストに抽出
        allPaperData.abstList.append(paper["abstract"])

        # 分類されたアブストラクトごとにリストに抽出
        labelAbst = labeledAbstDict[paper["title"]]
        for label in labelList:
            allPaperData.labelList[label].append(labelAbst[label])

    # 辞書をリストに変換
    allPaperData.paperList = list(allPaperData.paperDict.values())

    # テスト用のクエリ論文のインデックスを抽出
    for i, paper in enumerate(allPaperData.paperList):
        if paper['test'] == 1:
            testPaperData.allDataIndex.append(i)
            testPaperData.paperList.append(paper)
            allPaperData.testDataIndex.append(len(testPaperData.paperList) - 1)
        else:
            allPaperData.testDataIndex.append(None)

    # 予測結果の分析のため、タイトルをキーとして、indexをバリューとする辞書を生成
    for i, paper in enumerate(allPaperData.paperList):
        allPaperData.titleToIndex[paper['title']] = i

    correct_match = 0
    incorrect_match = 0

    correct_inclusion = 0
    incorrect_inclusion = 0

    for query_paper in testPaperData.paperList:
        if not "similar_label" in query_paper:
            continue
        for i, cited_paper_title in enumerate(query_paper["cite"]):
            query_paper_emb = labeledAbstDict[query_paper["title"]]
            cited_paper_emb = labeledAbstDict[cited_paper_title]

            # 精度計算
            if len(query_paper["similar_label"][i]) == 0:
                continue

            print('---------------------------------')
            print(query_paper["title"])
            print(cited_paper_title)

            # 一致判定
            result = hantei_match(query_paper_emb, cited_paper_emb, query_paper["similar_label"][i])
            if result:correct_match+=1
            else:incorrect_match+=1

            # 含有判定
            result = hantei_inclusion(query_paper_emb, cited_paper_emb, query_paper["similar_label"][i])
            if result:correct_inclusion+=1
            else:incorrect_inclusion+=1

    accuracy_match = correct_match / (correct_match + incorrect_match)
    accuracy_inclusion = correct_inclusion / (correct_inclusion + incorrect_inclusion)
    # print('================ result ================')
    # print(embedding_dir)
    # print(f"Correct: {correct_match}, Incorrect: {incorrect_match}")
    # print(f"Accuracy: {accuracy_match * 100:.2f}%")

    return accuracy_match, accuracy_inclusion

# 類似度計算関数
def calculate_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

# 最も類似する観点を特定する関数
def find_most_similar_aspect(query_paper, candidate_paper):
    aspects = ["bg", "obj", "method", "res"]
    similarity_scores = {}

    for aspect in aspects:
        if query_paper[aspect] != None and candidate_paper[aspect] != None:
            similarity_scores[aspect] = calculate_similarity(query_paper[aspect], candidate_paper[aspect])
    
    # romdom baseline
    # random_int =  random.randint(0, len(similarity_scores.keys())-1)
    # return list(similarity_scores.keys())[random_int]

    return max(similarity_scores, key=similarity_scores.get)

# 予測と実際のラベルの比較を行う関数
def hantei_match(query_paper, candidate_paper, sim_labels):
    predicted_aspect = find_most_similar_aspect(query_paper, candidate_paper)
    # print(query_paper)
    # print(candidate_paper)
    # print(f"predicted_aspect: {predicted_aspect}")
    # print(f"sim_label: {sim_labels[0]}")
    # print(f"sim_labels: {sim_labels}")

    # ケース1: 最も類似する観点に一致するか
    if predicted_aspect == sim_labels[0]:
        return True
    # ケース2: 類似する観点どれかに一致するか
    # if predicted_aspect in sim_labels:
    #     return True
    else:
        return False


# 予測と実際のラベルの比較を行う関数
def hantei_inclusion(query_paper, candidate_paper, sim_labels):
    predicted_aspect = find_most_similar_aspect(query_paper, candidate_paper)
    # print(query_paper)
    # print(candidate_paper)
    # print(f"predicted_aspect: {predicted_aspect}")
    # print(f"sim_label: {sim_labels[0]}")
    # print(f"sim_labels: {sim_labels}")

    # ケース1: 最も類似する観点に一致するか
    # if predicted_aspect == sim_labels[0]:
    #     return True
    # ケース2: 類似する観点どれかに一致するか
    if predicted_aspect in sim_labels:
        return True
    else:
        return False

if __name__ == "__main__":
    main()