import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.download_utils import Downloader


class StsbReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_stsb_raw_data(raw_data_path)
        self.train_file = 'train.tsv'
        self.dev_file = 'dev.tsv'
        self.test_file = 'test.tsv'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)

    def _read_train(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[0]
        contents = lines[1:]
        for line in contents:
            index, genre, filename, year, old_index, source1, source2, sentence1, sentence2, score = line.strip().split(
                "\t")
            datable("index", index)
            datable("genre", genre)
            datable("filename", filename)
            datable("year", year)
            datable("old_index", old_index)
            datable("source1", source1)
            datable("source2", source2)
            datable("sentence1", sentence1)
            datable("sentence2", sentence2)
            datable("score", score)

        return datable

    def _read_dev(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[1]
        contents = lines[1:]
        for line in contents:
            index, genre, filename, year, old_index, source1, source2, sentence1, sentence2, score = line.strip().split(
                "\t")
            datable("index", index)
            datable("genre", genre)
            datable("filename", filename)
            datable("year", year)
            datable("old_index", old_index)
            datable("source1", source1)
            datable("source2", source2)
            datable("sentence1", sentence1)
            datable("sentence2", sentence2)
            datable("score", score)

        return datable

    def _read_test(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[1]
        contents = lines[1:]
        for line in contents:
            index, genre, filename, year, old_index, source1, source2, sentence1, sentence2 = line.strip().split(
                "\t")
            datable("index", index)
            datable("genre", genre)
            datable("filename", filename)
            datable("year", year)
            datable("old_index", old_index)
            datable("source1", source1)
            datable("source2", source2)
            datable("sentence1", sentence1)
            datable("sentence2", sentence2)

        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        return None


if __name__ == "__main__":
    reader = StsbReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/STS_B/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
