import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import json
import ast

class NaturalQuestionsReader(BaseReader):
    def __init__(self, raw_data_path, use_cache=True):
        super(NaturalQuestionsReader, self).__init__()
        self.raw_data_path = raw_data_path
        self.use_cache = use_cache
        # downloader = Downloader()
        # downloader.download_commonsenseqa_raw_data(raw_data_path)
        self.train_file = 'nq-train.csv'
        self.dev_file = 'nq-dev.csv'
        self.test_file = 'nq-test.csv'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()
        self.addition = {}

    def _read_train(self, path):
        return self._read_data(path)

    def _read_dev(self, path):
        return self._read_data(path)

    def _read_test(self, path):
        return self._read_data(path)


    def _read_data(self, path):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            question,answers_text = line.strip('\n').split('\t')
            answers = ast.literal_eval(answers_text)
            datable("question",question)
            datable("answers",answers)
        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path) ,self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = NaturalQuestionsReader(
        raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    print("end")
