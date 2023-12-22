# allennlp dataloading packages
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from transformers import AutoTokenizer, AutoModel

# pytorch packages
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
import torch.nn as nn
import torch
import json
import pickle
from typing import Dict
import argparse
from argparse import Namespace
import glob
import random
import numpy as np
import itertools
import logging
from nltk.tokenize.treebank import TreebankWordDetokenizer

from nltk.tokenize import sent_tokenize

"""
SPECTERの学習データを取り出す
"""
def main():

    # インスタンスを作成
    reader = DataReaderFromPickled()

    # pklファイルのパス
    # file_path = "/workspace/dataserver/specterData/train.pkl"
    file_path = "/workspace/dataserver/specterData/val.pkl"


    # データを逐次読み込む
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/scibert_scivocab_cased")
    dataset = IterableDataSetMultiWorker(
        file_path=file_path, tokenizer=tokenizer, size=684100)


    # データを逐次読み込む
    triples = []
    paperDict = {}
    sscInputLineToTitle = []
    sscInput = []
    for source_input, pos_input, neg_input in dataset:
        

        # print('---------------------------------------------------------------')
        # print(source_input)
        # print(detokenize_and_split(source_input))
        # print(pos_input)
        # print(detokenize_and_split(pos_input))
        # print(neg_input)
        # print(detokenize_and_split(neg_input))

        triple = {}

        # source
        source_title, source_abst = detokenize_and_split(source_input)
        triple['source'] = source_title
        if not source_title in paperDict:
            paperDatum = {'title' : source_title, 'abstract': source_abst}
            paperDict[source_title] = paperDatum
            if len(sent_tokenize(source_abst, language='english')) > 0:
                datal = {'abstract_id': 0, 'sentences': sent_tokenize(source_abst, language='english')}
                sscInput.append(datal)
                sscInputLineToTitle.append(source_title)

        
        # pos
        pos_title, pos_abst = detokenize_and_split(pos_input)
        triple['pos'] = pos_title
        if not pos_title in paperDict:
            paperDatum = {'title' : pos_title, 'abstract': pos_abst}
            paperDict[pos_title] = paperDatum
            if len(sent_tokenize(pos_abst, language='english')) > 0:
                datal = {'abstract_id': 0, 'sentences': sent_tokenize(pos_abst, language='english')}
                sscInput.append(datal)
                sscInputLineToTitle.append(pos_title)
        
        # neg
        neg_title, neg_abst = detokenize_and_split(neg_input)
        triple['neg'] = neg_title
        if not neg_title in paperDict:
            paperDatum = {'title' : neg_title, 'abstract': neg_abst}
            paperDict[neg_title] = paperDatum
            if len(sent_tokenize(neg_abst, language='english')) > 0:
                datal = {'abstract_id': 0, 'sentences': sent_tokenize(neg_abst, language='english')}
                sscInput.append(datal)
                sscInputLineToTitle.append(neg_title)

        triples.append(triple)

        # print(len(triples))
    
    triplePath = './specter_triple-val.json'
    paperDictPath = './specter_paperDict.json'
    sscInputLineToTitlePath = './specter_ssc_input_line_to_title.json'
    sscInputPath = './specter_ssc_input.jsonl'

    with open(triplePath, 'w') as f:
        json.dump(triples, f, indent=4)    

    with open(paperDictPath, 'w') as f:
        json.dump(paperDict, f, indent=4)    

    with open(sscInputLineToTitlePath, 'w') as f:
        json.dump(sscInputLineToTitle, f, indent=4)    
    
    with open(sscInputPath, 'w') as f:
        for l in sscInput:
            f.write(json.dumps(l) + '\n')


class DataReaderFromPickled(DatasetReader):
    """
    pklファイルを読むためのクラス
    This is copied from https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/data.py#L61:L109 without any change
    """

    def __init__(self,
                 lazy: bool = False,
                 word_splitter: WordSplitter = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = 256,
                 concat_title_abstract: bool = None
                 ) -> None:
        """
        Dataset reader that uses pickled preprocessed instances
        Consumes the output resulting from data_utils/create_training_files.py

        the additional arguments are not used here and are for compatibility with
        the other data reader at prediction time
        """
        self.max_sequence_length = max_sequence_length
        self.token_indexers = token_indexers
        self._concat_title_abstract = concat_title_abstract
        super().__init__(lazy)

    def _read(self, file_path: str):
        """
        Args:
            file_path: path to the pickled instances
        """
        count = 0
        with open(file_path, 'rb') as f_in:
            unpickler = pickle.Unpickler(f_in)

            while True:
                try:
                    count += 1
                    # if count > 10:
                    #     break
                    instance = unpickler.load()
                    # compatibility with old models:
                    # for field in instance.fields:
                    #     if hasattr(instance.fields[field], '_token_indexers') and 'bert' in instance.fields[field]._token_indexers:
                    #         if not hasattr(instance.fields['source_title']._token_indexers['bert'], '_truncate_long_sequences'):
                    #             instance.fields[field]._token_indexers['bert']._truncate_long_sequences = True
                    #             instance.fields[field]._token_indexers['bert']._token_min_padding_length = 0
                    # print(instance)
                    # exit()
                    if self.max_sequence_length:
                        for paper_type in ['source', 'pos', 'neg']:
                            if self._concat_title_abstract:
                                tokens = []
                                title_field = instance.fields.get(
                                    f'{paper_type}_title')
                                abst_field = instance.fields.get(
                                    f'{paper_type}_abstract')
                                if title_field:
                                    tokens.extend(title_field.tokens)
                                if tokens:
                                    tokens.extend([Token('[SEP]')])
                                if abst_field:
                                    tokens.extend(abst_field.tokens)
                                if title_field:
                                    title_field.tokens = tokens
                                    instance.fields[f'{paper_type}_title'] = title_field
                                elif abst_field:
                                    abst_field.tokens = tokens
                                    instance.fields[f'{paper_type}_title'] = abst_field
                                else:
                                    yield None
                                # title_tokens = get_text_tokens(query_title_tokens, query_abstract_tokens, abstract_delimiter)
                                # pos_title_tokens = get_text_tokens(pos_title_tokens, pos_abstract_tokens, abstract_delimiter)
                                # neg_title_tokens = get_text_tokens(neg_title_tokens, neg_abstract_tokens, abstract_delimiter)
                                # query_abstract_tokens = pos_abstract_tokens = neg_abstract_tokens = []
                            for field_type in ['title', 'abstract', 'authors', 'author_positions']:
                                field = paper_type + '_' + field_type
                                if instance.fields.get(field):
                                    instance.fields[field].tokens = instance.fields[field].tokens[
                                        :self.max_sequence_length]
                                    # print(instance.fields[field].tokens[
                                    #     :self.max_sequence_length])
                                if field_type == 'abstract' and self._concat_title_abstract:
                                    instance.fields.pop(field, None)
                    # print("Data Count: ", count)
                    yield instance
                except EOFError:

                    break


class IterableDataSetMultiWorker(IterableDataset):
    def __init__(self, file_path, tokenizer, size, block_size=100):
        self.datareaderfp = DataReaderFromPickled(max_sequence_length=512)
        self.data_instances = self.datareaderfp._read(file_path)

        self.tokenizer = tokenizer
        self.size = size
        self.block_size = block_size

    def __iter__(self):
        iter_end = self.size
        for data_instance in itertools.islice(self.data_instances, iter_end):
            data_input = self.ai2_to_transformers(
                data_instance, self.tokenizer)
            # print(data_input)
            yield data_input

        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:
        #     iter_end = self.size
        #     for data_instance in itertools.islice(self.data_instances, iter_end):
        #         data_input = self.ai2_to_transformers(
        #             data_instance, self.tokenizer)
        #         yield data_input

        # else:
        #     # when num_worker is greater than 1. we implement multiple process data loading.
        #     iter_end = self.size
        #     worker_id = worker_info.id
        #     num_workers = worker_info.num_workers
        #     i = 0
        #     for data_instance in itertools.islice(self.data_instances, iter_end):
        #         if int(i / self.block_size) % num_workers != worker_id:
        #             i = i + 1
        #             pass
        #         else:
        #             i = i + 1
        #             data_input = self.ai2_to_transformers(
        #                 data_instance, self.tokenizer)
        #             yield data_input

    def ai2_to_transformers(self, data_instance, tokenizer):
        """
        Args:
            data_instance: ai2 data instance
            tokenizer: huggingface transformers tokenizer
        """
        source_tokens = data_instance["source_title"].tokens
        # for key, value in data_instance["source_title"].__dict__.items():
        #     print(key, ':', value)
        # exit()
        source_title = tokenizer(' '.join([str(token) for token in source_tokens]),
                                 truncation=True, padding="max_length", return_tensors="pt",
                                 max_length=512)
        source_input = {'input_ids': source_title['input_ids'][0],
                        'token_type_ids': source_title['token_type_ids'][0],
                        'attention_mask': source_title['attention_mask'][0]}

        pos_tokens = data_instance["pos_title"].tokens
        pos_title = tokenizer(' '.join([str(token) for token in pos_tokens]),
                              truncation=True, padding="max_length", return_tensors="pt", max_length=512)

        pos_input = {'input_ids': pos_title['input_ids'][0],
                     'token_type_ids': pos_title['token_type_ids'][0],
                     'attention_mask': pos_title['attention_mask'][0]}

        neg_tokens = data_instance["neg_title"].tokens
        neg_title = tokenizer(' '.join([str(token) for token in neg_tokens]),
                              truncation=True, padding="max_length", return_tensors="pt", max_length=512)

        neg_input = {'input_ids': neg_title['input_ids'][0],
                     'token_type_ids': neg_title['token_type_ids'][0],
                     'attention_mask': neg_title['attention_mask'][0]}

        # return source_input, pos_input, neg_input
        return source_tokens, pos_tokens, neg_tokens


def detokenize_and_split(tokens):
    sentence1 = ''
    sentence2 = ''
    current_sentence = 1

    for i, token in enumerate(tokens):
        token_text = token.text.strip()  # Get the text attribute from the Token object

        if token_text == '[SEP]':
            current_sentence = 2
            continue
        
        prev_prev_token = tokens[i-2].text if i > 1 else None
        prev_token = tokens[i-1].text if i > 0 else None
        next_token = tokens[i+1].text if i < len(tokens) - 1 else None

        # Special handling for periods between numbers (e.g., "1.22")
        # if token_text == '.' and prev_token and next_token and prev_token.isdigit() and next_token.isdigit():
        if token_text == '.' or token_text == ',':
            separator = ''
        elif token_text.isdigit() and prev_token == '.' and \
            prev_prev_token and (prev_prev_token.isdigit() or in_num(prev_prev_token)):
            separator = ''
        # Special handling for certain symbols that should not have spaces around them
        # elif token_text in ['-', '/', '(', ')', '"', "'", '$', '<=', '>=', '!=', '%'] or \
        #         prev_token in ['-', '/', '(', '"', "'", '$', '<=', '>=', '!=']:
        elif token_text in ['-', '/', ')', '$', '<=', '>=', '!=', '%'] or \
            prev_token in ['-', '/', '(', '$', '<=', '>=', '!=']:
            separator = ''
        else:
            separator = ' '

        if current_sentence == 1:
            sentence1 += separator + token_text
        else:
            sentence2 += separator + token_text

    # Remove the trailing period from sentence1 if it exists
    if sentence1.endswith('.'):
        sentence1 = sentence1[:-1]

    return sentence1.strip(), sentence2.strip()

def in_num(s):
    for i in range(len(s)):
        if s[i].isdigit():
            return True
    
    return False


if __name__ == '__main__':
    main()
