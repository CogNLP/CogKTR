import os
from cogktr.data.reader.basereader import BaseReader
from cogktr.data.datable import DataTable


class SST2Reader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_file = 'train.tsv'
        self.dev_file = 'dev.tsv'
        self.test_file = 'test.tsv'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)

    def _read(self, path):
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header=lines[0]
        contents=lines[1:]
        for line in contents:
            text,label=line.strip().split("\t")
            datable("text", text)
            datable("label", label)
        return datable


    def read_train(self):
        return self._read(self.train_path)

    def read_dev(self):
        return self._read(self.dev_path)

    def read_test(self):
        return self._read(self.test_path)

    def read_all(self):
        return self.read_train(), self.read_dev(), self.read_test()

if __name__=="__main__":
    reader=SST2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data,dev_data,test_data=reader.read_all()
    print("end")
