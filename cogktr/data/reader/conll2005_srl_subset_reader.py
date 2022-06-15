import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import json
from cogktr.utils.constant.conll2005_srl_subset_constant.vocab import *


class Conll2005SrlSubsetReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_conll2005_srl_subset_raw_data(raw_data_path)
        self.train_file = 'train.json'
        self.dev_file = 'dev.json'
        self.test_file = 'wsj-test.json'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.pos_label_vocab = Vocabulary()
        self.dep_label_vocab = Vocabulary()
        self.tag_label_vocab = Vocabulary()
        self.tag_label_vocab.add_dict(CoNLL2005_SRL_LABEL_TO_ID)
        # TODO: add complete vocabulary

    def _read_data(self, path):
        datable = DataTable()
        file = open(path, 'r', encoding='utf-8')
        lines = []
        for line in file.readlines():
            dic = json.loads(line)
            lines.append(dic)
        for line in lines:
            sentence_id = line["sentence_id"]
            tokens = line["tokens"]
            pos_tags = line["pos_tags"]
            verb_indicator = line["verb_indicator"]
            dep_head = line["dep_head"]
            dep_label = line["dep_label"]
            tags = line["tags"]
            metadata = line["metadata"]
            datable("sentence_id", sentence_id)
            datable("tokens", tokens)
            datable("pos_tags", pos_tags)
            datable("verb_indicator", verb_indicator)
            datable("dep_head ", dep_head)
            datable("dep_label", dep_label)
            datable("tags", tags)
            datable(" metadata", metadata)

        return datable

    def read_all(self):
        return self._read_data(self.train_path), self._read_data(self.dev_path), self._read_data(self.test_path)

    def read_vocab(self):
        self.pos_label_vocab.create()
        self.dep_label_vocab.create()
        self.tag_label_vocab.create()
        return {"pos_label_vocab": self.pos_label_vocab,
                "dep_label_vocab": self.dep_label_vocab,
                "tag_label_vocab": self.tag_label_vocab}


if __name__ == "__main__":
    reader = Conll2005SrlSubsetReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2005_srl_subset/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
