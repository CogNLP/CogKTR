import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import json


class NaturalQuestionsReader(BaseReader):
    def __init__(self, raw_data_path, use_cache=True):
        super(NaturalQuestionsReader, self).__init__()
        self.raw_data_path = raw_data_path
        self.use_cache = use_cache
        # downloader = Downloader()
        # downloader.download_commonsenseqa_raw_data(raw_data_path)
        self.train_file = 'nq-train.json'
        self.dev_file = 'nq-dev.json'
        self.test_file = None
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = None
        self.label_vocab = Vocabulary()
        self.addition = {}

    def _read_train(self, path):
        return self._read_data(path)

    def _read_dev(self, path):
        return self._read_data(path)


    def _read_data(self, path):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = json.load(file)
        print("Debug Usage")
        for single_data in lines:
            datable("question",single_data["question"])
            datable("answers", single_data["answers"])
            datable("positive_ctxs", single_data["positive_ctxs"])
            datable("negative_ctxs", single_data["negative_ctxs"])
            datable("hard_negative_ctxs", single_data["hard_negative_ctxs"])
        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path) ,None

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = NaturalQuestionsReader(
        raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    print("end")
