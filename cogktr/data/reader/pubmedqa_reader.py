import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import json


class PubMedQAReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        # downloader = Downloader()
        # downloader.download_lama_raw_data(raw_data_path)
        self.label_vocab = Vocabulary()

    def _read_data(self):
        datable = DataTable()
        with open(os.path.join(self.raw_data_path, "ori_pqaa.json")) as file:
            items = json.load(file)
        for _, value in items.items():
            question = value["QUESTION"]
            answer = value["final_decision"]
            datable("question", question)
            datable("answer", answer)
            self.label_vocab.add(answer)
        return datable

    def read_all(self, isSplit=True):
        if isSplit:
            return self._read_data().split(0.8, 0.1, 0.1)
        else:
            return self._read_data()

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = PubMedQAReader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/question_answering/PubMedQA/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
