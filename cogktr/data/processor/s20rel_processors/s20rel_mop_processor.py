from cogktr.data.processor.base_processor import BaseProcessor
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import AutoTokenizer
from transformers import BertTokenizer
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

from cogktr.data.reader.s20rel_reader import S20relReader


class S20relMopProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        # self.vocab = vocab
        self.tokenizer = AutoTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        for text_head, text_relation, label in tqdm(zip(data['text_head'], data['text_relation'], data['label']),
                                                    total=len(data['label']),
                                                    desc="Processing data..."):
            text_feature = self.tokenizer.batch_encode_plus(
                [text_head, text_relation],
                padding="max_length",  # First sentence will have some PADDED tokens to match second sequence length
                max_length=self.max_token_len,
                return_tensors="pt",
                truncation=True,
            )
            datable("input_ids", text_feature.data["input_ids"])
            datable("token_type_ids", text_feature.data["token_type_ids"])
            datable("attention_mask", text_feature.data["attention_mask"])
            datable("label", data["label"])
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

if __name__ == "__main__":
    s20rel_reader = S20relReader("/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/S20Rel")
    train_data = s20rel_reader.read_train(group_idx=0)
    vocab = s20rel_reader.read_vocab()

    s20rel_processor = S20relMopProcessor(plm="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                          max_token_len= 128)
    train_dataset = s20rel_processor.process_train(train_data)
    print("end")