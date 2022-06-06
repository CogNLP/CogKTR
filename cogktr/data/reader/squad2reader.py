import os
from cogktr.data.reader.basereader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.io_utils import load_json
from cogktr.utils.download_utils import Downloader


class SQUAD2Reader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_squad2_raw_data(raw_data_path)
        self.train_file = 'train-v2.0.json'
        self.dev_file = 'dev-v2.0.json'
        self.test_file = ''
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)

    def _read_train(self, path):
        datable = DataTable()
        raw_data_dict = load_json(path)
        data = raw_data_dict["data"]
        for line in data:
            title = line["title"]
            paragraphs = line["paragraphs"]
            datable("title", title)
            datable("paragraphs ", paragraphs)

        return datable

    def _read_dev(self, path):
        datable = DataTable()
        raw_data_dict = load_json(path)
        data = raw_data_dict["data"]
        for line in data:
            title = line["title"]
            paragraphs = line["paragraphs"]
            datable("title", title)
            datable("paragraphs ", paragraphs)

        return datable

    def _read_test(self, path):
        return None

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        return None


if __name__ == "__main__":
    reader = SQUAD2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
