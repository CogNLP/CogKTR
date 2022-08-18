import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import json


class CommonsenseqaForSAFEReader(BaseReader):
    def __init__(self, raw_data_path, use_cache=True):
        super(CommonsenseqaForSAFEReader, self).__init__()
        self.raw_data_path = raw_data_path
        self.use_cache = use_cache
        downloader = Downloader()
        downloader.download_commonsenseqa_raw_data(raw_data_path)
        self.train_file = 'train.statement.jsonl'
        self.dev_file = 'dev.statement.jsonl'
        self.test_file = 'test.statement.jsonl'
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
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            json_line = json.loads(line)
            for idx,choice in enumerate(json_line["question"]["choices"]):
                datable("id", json_line["id"])
                datable("stem",json_line["question"]["stem"])
                datable("answer_text",choice["text"])
                datable("key",choice["label"])
                datable("statement",json_line["statements"][idx]["statement"])
        return datable


    def _read_data(self, path):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            json_line = json.loads(line)
            for idx,choice in enumerate(json_line["question"]["choices"]):
                datable("id", json_line["id"])
                datable("stem",json_line["question"]["stem"])
                datable("answer_text",choice["text"])
                datable("key",choice["label"])
                datable("statement",json_line["statements"][idx]["statement"])
                datable("answerKey",json_line["answerKey"])
            self.label_vocab.add(json_line["answerKey"])
        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


class CommonsenseqaReader(BaseReader):
    def __init__(self, raw_data_path, use_cache=True):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.use_cache = use_cache
        downloader = Downloader()
        downloader.download_commonsenseqa_raw_data(raw_data_path)
        self.train_file = 'train.statement.jsonl'
        self.dev_file = 'dev.statement.jsonl'
        self.test_file = 'test.statement.jsonl'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()
        self.addition = {}

    def _read_data(self, path):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            line_dict = json.loads(line)
            example_id = line_dict["id"]
            answerkey = line_dict["answerKey"]
            question = line_dict["question"]
            statements = line_dict["statements"]
            datable("example_id", example_id)
            datable("answerkey", answerkey)
            datable("question", question)
            datable("statements", statements)
            self.label_vocab.add(answerkey)
        return datable

    def _read_train(self, path):
        return self._read_data(path)

    def _read_dev(self, path):
        return self._read_data(path)

    def _read_test(self, path, adj_path=None, adj_cache_path=None):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            line_dict = json.loads(line)
            example_id = line_dict["id"]
            question = line_dict["question"]
            statements = line_dict["statements"]
            datable("example_id", example_id)
            datable("question", question)
            datable("statements", statements)
        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = CommonsenseqaForSAFEReader(
        raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/CommonsenseQA/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
