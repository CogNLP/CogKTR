import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import json
import csv


class MultisegchnsentibertReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_multisegchnsentibert_raw_data(raw_data_path)
        self.train_file = 'train.tsv'
        self.dev_file = 'dev.tsv'
        self.test_file = 'test.tsv'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()

    def _read_data(self, path):
        datable = DataTable()
        with open(path, encoding='utf-8') as file:
            lines = csv.reader(file, delimiter="\t")
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                sentence = json.loads(line[0])
                thulac_word_length = json.loads(line[1])
                nlpir_word_length = json.loads(line[2])
                hanlp_word_length = json.loads(line[3])
                label = json.loads(line[4])
                datable("sentence", sentence)
                datable("thulac_word_length", thulac_word_length)
                datable("nlpir_word_length", nlpir_word_length)
                datable("hanlp_word_length", hanlp_word_length)
                datable("label", label)
                self.label_vocab.add(label)
        return datable

    def read_all(self):
        return self._read_data(self.train_path), self._read_data(self.dev_path), self._read_data(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = MultisegchnsentibertReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/MultiSegChnSentiBERT/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
