import os
from cogktr.data.reader.basereader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary


class QNLIReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_file = 'train.tsv'
        self.dev_file = 'dev.tsv'
        self.test_file = 'test.tsv'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()

    def _read_data(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[0]
        contents = lines[1:]
        for line in contents:
            index,question,sentence,label= line.strip().split("\t")
            datable("index", index)
            datable("question", question)
            datable("sentence", sentence)
            datable("label", label)
            self.label_vocab.add(label)

        return datable

    def _read_train(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[0]
        contents = lines[1:]
        for line in contents:
            index, question, sentence, label = line.strip().split("\t")
            datable("index", index)
            datable("question", question)
            datable("sentence", sentence)
            datable("label", label)
            self.label_vocab.add(label)

        return datable
    def _read_dev(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[0]
        contents = lines[1:]
        for line in contents:
            index, question, sentence, label = line.strip().split("\t")
            datable("index", index)
            datable("question", question)
            datable("sentence", sentence)
            datable("label", label)
            self.label_vocab.add(label)

        return datable

    def _read_test(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[0]
        contents = lines[1:]
        for line in contents:
            index, question, sentence = line.strip().split("\t")
            datable("index", index)
            datable("question", question)
            datable("sentence", sentence)

        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = QNLIReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
