from torch.utils.data import Dataset
from torch import tensor
import re
import traceback


class MyDataset(Dataset):
    """観点の情報が付与された論文推薦データセットを扱うクラス

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, data, paper_dict, ssc_result_dict, label_dict, tokenizer=None, block_size=100):
        self.data_instances = data
        self.paper_dict = paper_dict
        self.ssc_result_dict = ssc_result_dict
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.label_dict = label_dict

    def __len__(self):
        return len(self.data_instances)

    def __getitem__(self, index):
        data_instance = self.data_instances[index]
        return self.retTransformersInput(data_instance, self.tokenizer)

    def retTransformersInput(self, data_instance, tokenizer):
        # print("data_instance: ", data_instance)
        source_title = re.sub(r'\s+', ' ', data_instance["source"])
        pos_title = re.sub(r'\s+', ' ', data_instance["pos"])
        neg_title = re.sub(r'\s+', ' ', data_instance["neg"])

        # debug
        # print(f'source_title: {source_title}')
        # print(f'source_abst: {self.paper_dict[source_title]["abstract"]}')

        source_title_abs = self.paper_dict[source_title]["title"] + \
            tokenizer.sep_token + \
            self.paper_dict[source_title]["abstract"]
        sourceEncoded = self.tokenizer(
            source_title_abs,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        pos_title_abs = self.paper_dict[pos_title]["title"] + \
            tokenizer.sep_token + \
            self.paper_dict[pos_title]["abstract"]
        posEncoded = self.tokenizer(
            pos_title_abs,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        neg_title_abs = self.paper_dict[neg_title]["title"] + \
            tokenizer.sep_token + \
            self.paper_dict[neg_title]["abstract"]
        negEncoded = self.tokenizer(
            neg_title_abs,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        source_input = {
            'input_ids': sourceEncoded["input_ids"][0],
            'attention_mask': sourceEncoded["attention_mask"][0],
            'token_type_ids': sourceEncoded["token_type_ids"][0]
        }
        pos_input = {
            'input_ids': posEncoded["input_ids"][0],
            'attention_mask': posEncoded["attention_mask"][0],
            'token_type_ids': posEncoded["token_type_ids"][0]
        }
        neg_input = {
            'input_ids': negEncoded["input_ids"][0],
            'attention_mask': negEncoded["attention_mask"][0],
            'token_type_ids': negEncoded["token_type_ids"][0]
        }

        # SPECTER データで本文が無くてSSCができなかったケースに対応
        if not source_title in self.ssc_result_dict:
            # titleのみラベルポジションを付与
            # ex. [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,....]
            source_position_label = self.tokenize_with_label_title_only(
                source_title_abs, self.tokenizer)
        else:
            source_position_label = self.tokenize_with_label(
                source_title_abs, self.ssc_result_dict[source_title], self.tokenizer)
        if not pos_title in self.ssc_result_dict:
            pos_position_label = self.tokenize_with_label_title_only(
                pos_title_abs, self.tokenizer)
        else:
            pos_position_label = self.tokenize_with_label(
                pos_title_abs, self.ssc_result_dict[pos_title], self.tokenizer)
        if not neg_title in self.ssc_result_dict:
            neg_position_label = self.tokenize_with_label_title_only(
                neg_title_abs, self.tokenizer)
        else:
            neg_position_label = self.tokenize_with_label(
                neg_title_abs, self.ssc_result_dict[neg_title], self.tokenizer)

        ret_source = {
            "input": source_input,
            "position_label_list": tensor(source_position_label[1])
        }
        ret_pos = {
            "input": pos_input,
            "position_label_list": tensor(pos_position_label[1])
        }
        ret_neg = {
            "input": neg_input,
            "position_label_list": tensor(neg_position_label[1])
        }

        print(source_title,', \n'  , pos_title,', \n' , neg_title,', \n' )
        return ret_source, ret_pos, ret_neg

    def tokenize_with_label(self, sentence, ssc_result, tokenizer):
        tokenized_input = tokenizer(
            sentence,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # input['input_ids'][0](トークナイズされた入力文)と同じ長さで、その位置のトークンのラベルを格納するリスト
        # label_list_for_words = [None for i in range(len(
        #     tokenized_input['input_ids'][0].tolist()))]

        # 最大長に合わせる
        # 【注意】 0ではなくNoneを入れていたところデータローダーの制約に引っかかってエラーとなる
        label_list_for_words = [0 for i in range(512)]

        # titleの位置にラベルを格納する
        # [SEP]の位置を特定する
        sep_pos = tokenized_input['input_ids'][0].tolist().index(tokenizer.sep_token_id)
        for i in range(1, sep_pos):
            # label_list_for_words[i] = 'title'
            label_list_for_words[i] = 1

        # [SEP]と[CLS]は-1
        label_list_for_words[0] = -1
        label_list_for_words[sep_pos] = -1
        try:
            # 各単語の観点をlabel_positionsに格納
            for text_label_pair in ssc_result:
                text = text_label_pair[0]
                label = self.label_dict[text_label_pair[1]]

                # 1文単位でtokenizeする
                tokenizedText = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512
                )
                # tokenizeされたテキストから先頭の[CLS]と末尾の[SEP]を取り除く
                tokenizedText_input_ids = tokenizedText['input_ids'][0][1:-1].tolist()

                start, end = self.find_subarray(
                    tokenized_input['input_ids'][0].tolist(), tokenizedText_input_ids)
                # たまに見つからないときがある（なぜ？）
                if start and end:
                    for i in range(start, end+1):
                        label_list_for_words[i] = label

            return tokenized_input, label_list_for_words

        except Exception as e:
            print("text: ", text)
            print("tokenized_input['input_ids'][0].tolist()",
                  tokenized_input['input_ids'][0].tolist())
            print("tokenized_input_ids", tokenizedText_input_ids)
            print(traceback.format_exc())

    def find_subarray(self, arr, subarr):
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
        return (None, None)
    
    def tokenize_with_label_title_only(self, sentence, tokenizer):
        tokenized_input = tokenizer(
            sentence,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # 最大長に合わせる
        label_list_for_words = [0 for i in range(512)]

        # titleの位置にラベルを格納する
        # [SEP]の位置を特定する
        sep_pos = tokenized_input['input_ids'][0].tolist().index(tokenizer.cls_token_id)
        for i in range(1, sep_pos):
            # label_list_for_words[i] = 'title'
            label_list_for_words[i] = 1

        # [SEP]と[CLS]は-1
        label_list_for_words[0] = -1
        label_list_for_words[sep_pos] = -1
        return tokenized_input, label_list_for_words
    

class PredSscDataset(MyDataset):
    def __init__(self, data, paper_dict, ssc_result_dict, label_dict, tokenizer=None, block_size=100):
        super().__init__(data, paper_dict, ssc_result_dict, label_dict, tokenizer, block_size)

    def tokenize_with_label(self, sentence, ssc_result, tokenizer):
        tokenized_input = tokenizer(
            sentence,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # input['input_ids'][0](トークナイズされた入力文)と同じ長さで、その位置のトークンのラベルを格納するリスト
        # label_list_for_words = [None for i in range(len(
        #     tokenized_input['input_ids'][0].tolist()))]

        # 最大長に合わせる
        # 【注意】 0ではなくNoneを入れていたところデータローダーの制約に引っかかってエラーとなる
        label_list_for_words = [0 for i in range(512)]

        # titleの位置にラベルを格納する
        # [SEP]の位置を特定する
        sep_pos = tokenized_input['input_ids'][0].tolist().index(tokenizer.sep_token_id)
        for i in range(1, sep_pos):
            # label_list_for_words[i] = 'title'
            label_list_for_words[i] = 1

        # [SEP]と[CLS]は-1
        label_list_for_words[0] = -1
        label_list_for_words[sep_pos] = -1
        try:
            # 各単語の観点をlabel_positionsに格納
            for text_label_pair in ssc_result:
                text = text_label_pair[0]
                labelPreds = text_label_pair[1]
                print(labelPreds)
                exit()

                # 1文単位でtokenizeする
                tokenizedText = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512
                )
                # tokenizeされたテキストから先頭の[CLS]と末尾の[SEP]を取り除く
                tokenizedText_input_ids = tokenizedText['input_ids'][0][1:-1].tolist()

                start, end = self.find_subarray(
                    tokenized_input['input_ids'][0].tolist(), tokenizedText_input_ids)
                # たまに見つからないときがある（なぜ？）
                if start and end:
                    for i in range(start, end+1):
                        label_list_for_words[i] = label

            return tokenized_input, label_list_for_words

        except Exception as e:
            print("text: ", text)
            print("tokenized_input['input_ids'][0].tolist()",
                  tokenized_input['input_ids'][0].tolist())
            print("tokenized_input_ids", tokenizedText_input_ids)
            print(traceback.format_exc())