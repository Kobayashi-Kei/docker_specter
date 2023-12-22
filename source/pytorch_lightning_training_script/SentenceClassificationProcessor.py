import json


class SentenceClassificationProcessor:
    def __init__(self, input_path, output_path, tokenizer):
        # SSCの結果をロード & 整形
        with open(input_path, 'r') as f:
            ssc_input = f.readlines()
        with open(output_path, 'r') as f:
            ssc_output = f.readlines()

        ssc_paper_dict = {}
        for i, line in enumerate(ssc_output):
            line = json.loads(line)
            label = line[0]
            text_list = line[1]

            inputLine = json.loads(ssc_input[i])
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

            ssc_paper_dict[title] = text_list

        self.ssc_paper_dict = ssc_paper_dict
        self.tokenizer = tokenizer
        self.label_list = ["title", "bg", "obj", "method", "res"]

    def get_paper_ssc(self, title):
        return self.ssc_paper_dict[title]

    def find_sep_index(self, input_id_list: list) -> int:
        return input_id_list.index(102)

    def assign_title_label(self, text_label_list: list, sep_index: int):
        for i in range(1, sep_index):
            text_label_list[i] = 'title'

        return text_label_list

    def assign_sentence_label(self, text_label_list: list, paper_ssc: list):
        """
        観点への分類結果をもとに，tokernizerでトークナイズされた各単語に
        対応するラベルを格納する配列を返す．
        例. ["title", "title", ...,  "title", "bg", ... , "bg","method", ... ,"method]

        前提として，text_label_listには[SEP]より前の"title"のラベルが格納されているものとする．

        Args:
            text_label_list (list): Title [SEP] Abstract をトークナイズした各トークンに対応する観点を格納するリスト 
            paper_ssc (list): アブスト分類の結果
        """
        for text_label_pair in paper_ssc:
            text = text_label_pair[0]
            label = text_label_pair[1]

            # 1文単位でtokenizeする
            tokenizedText = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512
            )

            # 先頭の101([CLS])と末尾の102([SEP])を取り除く
            tokenizedText_input_ids = tokenizedText['input_ids'][0][1:-1].tolist()

            start, end = self.find_subarray(
                input['input_ids'][0].tolist(), tokenizedText_input_ids)
            for i in range(start, end+1):
                text_label_list[i] = label

        return text_label_list

    def merge_vectors_by_aspect(self, output: list, text_label_list: list):
        label_last_hideen_state = {label: [] for label in self.label_list}
        for i, tensor in enumerate(output):
            # [CLS]or [SEP]の時
            if text_label_list[i] == None or text_label_list[i] == "other":
                continue

            label_last_hideen_state[text_label_list[i]].append(tensor.tolist())

        return label_last_hideen_state

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
        return None
