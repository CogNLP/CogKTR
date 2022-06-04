import torch

from cogktr.data.reader.sst2reader import SST2Reader
from cogktr.enhancers import BaseEnhancer
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizerFast
from tqdm import tqdm
import numpy as np


class SST24KGEMBProcessor:
    def __init__(self, plm, max_token_len, vocab, enhancer):
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.enhancer = enhancer
        self.tokenizer = BertTokenizerFast.from_pretrained(plm)  # Not BertTokenizer

    def process_train(self, data):
        datable = DataTable()
        print("Processing data...")
        num = 0  # TODO:删掉
        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            num = num + 1  # TODO:删掉
            if num < 50:  # TODO:删掉
                output = self.tokenizer.encode_plus(text=sentence,
                                                    truncation=True,
                                                    padding="max_length",
                                                    add_special_tokens=True,
                                                    return_offsets_mapping=True,
                                                    max_length=self.max_token_len)
                token = output.data['input_ids']
                attention_mask = output.data['attention_mask']
                offset_mapping = output.data['offset_mapping']
                wikipedia = self.enhancer.get_knowledge(sentence)["wikipedia"]
                entity_mask_list = []
                entity_embedding_list = []
                entity_mask = np.zeros((len(offset_mapping), 1))
                entity_embedding = np.zeros((1, 100))
                for item in wikipedia:
                    entity_mask = np.zeros((len(offset_mapping), 1))
                    start_flag = 0
                    for index, (start, end) in enumerate(offset_mapping):
                        if item["begin"] >= start and item["begin"] <= end:
                            start_flag = 1
                        if item["end"] >= start and item["end"] <= end:
                            start_flag = 0
                            entity_mask[index] = 1
                        if start_flag == 1:
                            entity_mask[index] = 1
                        if offset_mapping[index] == (0, 0) and offset_mapping[index + 1] == (0, 0) and index < len(
                                offset_mapping):
                            break
                    entity_embedding = item["emb"]["entity_embedding"].reshape(1, -1)
                    entity_mask_list.append(entity_mask)
                    entity_embedding_list.append(entity_embedding)
                if len(entity_mask_list) == 0:
                    entity_mask_list.append(entity_mask)
                    entity_embedding_list.append(entity_embedding)
                datable("entity_mask_list", entity_mask_list)
                datable("entity_embedding_list", entity_embedding_list)
                datable("token", token)
                datable("attention_mask", attention_mask)
                datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_dev(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
                                          max_length=self.max_token_len)
            datable("wikipedia", self.enhancer.get_knowledge(sentence)["wikipedia"])
            datable("token", token)
            datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_test(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence in tqdm(zip(data['sentence']), total=len(data['sentence'])):
            token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
                                          max_length=self.max_token_len)
            datable("wikipedia", self.enhancer.get_knowledge(sentence)["wikipedia"])
            datable("token", token)
        return DataTableSet(datable)


if __name__ == "__main__":
    reader = SST2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    enhancer = Enhancer(return_entity_ebd=True)
    enhancer.set_config(
        WikipediaSearcherPath="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia/raw_data/entity.jsonl",
        WikipediaEmbedderPath="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl")
    processor = SST24KGEMBProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab, enhancer=enhancer)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
