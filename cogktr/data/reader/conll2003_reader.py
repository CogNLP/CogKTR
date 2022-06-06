import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader


class Conll2003Reader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_conll2003_raw_data(raw_data_path)
        self.train_file = 'train.txt'
        self.dev_file = 'valid.txt'
        self.test_file = 'test.txt'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.pos_label_vocab = Vocabulary()
        self.syn_label_vocab = Vocabulary()
        self.ner_label_vocab = Vocabulary()
        self.pos_label_vocab.add_dict({"<pad>": 0})
        self.syn_label_vocab.add_dict({"<pad>": 0})
        self.ner_label_vocab.add_dict({"<pad>": 0})

    def _read_data(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[:2]
        contents = lines[2:]
        sentence = list()
        pos_labels = list()
        syn_labels = list()
        ner_labels = list()
        for line in contents:
            if line != "\n":
                word, pos_label, syn_label, ner_label = line.strip().split(" ")
                sentence.append(word)
                pos_labels.append(pos_label)
                syn_labels.append(syn_label)
                ner_labels.append(ner_label)
                self.pos_label_vocab.add(pos_label)
                self.syn_label_vocab.add(syn_label)
                self.ner_label_vocab.add(ner_label)
            elif line == "\n":
                datable("sentence", sentence)
                datable("pos_labels", pos_labels)
                datable("syn_labels", syn_labels)
                datable("ner_labels", ner_labels)
                sentence = list()
                pos_labels = list()
                syn_labels = list()
                ner_labels = list()

        return datable

    def read_all(self):
        return self._read_data(self.train_path), self._read_data(self.dev_path), self._read_data(self.test_path)

    def read_vocab(self):
        self.pos_label_vocab.create()
        self.syn_label_vocab.create()
        self.ner_label_vocab.create()
        return {"pos_label_vocab": self.pos_label_vocab,
                "syn_label_vocab": self.syn_label_vocab,
                "ner_label_vocab": self.ner_label_vocab}


if __name__ == "__main__":
    reader = Conll2003Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2003/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
