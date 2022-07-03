import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import numpy as np
import json


class Commonsenseqa_Qagnn_Reader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_commonsenseqa_qagnn_raw_data(raw_data_path)
        self.train_file = 'statement/train.statement.jsonl'
        self.dev_file = 'statement/dev.statement.jsonl'
        self.test_file = 'statement/test.statement.jsonl'
        self.train_adj_file = 'graph/train.graph.adj.pk'
        self.dev_adj_file = 'graph/dev.graph.adj.pk'
        self.test_adj_file = 'graph/test.graph.adj.pk'
        self.ent_emb_file = 'cpnet/tzw.ent.npy'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.train_adj_path = os.path.join(raw_data_path, self.train_adj_file)
        self.dev_adj_path = os.path.join(raw_data_path, self.dev_adj_file)
        self.test_adj_path = os.path.join(raw_data_path, self.test_adj_file)
        self.ent_emb_path = os.path.join(raw_data_path, self.ent_emb_file)
        self.label_vocab = Vocabulary()
        self.addition = {}

    def _read_data(self, path, adj_path=None):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            line_dict = json.loads(line)
            example_id = line_dict["id"]
            answer_label = line_dict["answerKey"]
            candidate_character_list = [label_dict["label"] for label_dict in line_dict["question"]["choices"]]
            candidate_text_list = [label_dict["text"] for label_dict in line_dict["question"]["choices"]]
            candidate_num = len(candidate_text_list)
            context = [line_dict["question"]["stem"]] * candidate_num
            datable("example_id", example_id)
            datable("answer_label", answer_label)
            datable("context", context)
            datable("candidate_text_list", candidate_text_list)
            self.label_vocab.add_sequence(candidate_character_list)
        return datable

    def _read_train(self, path, adj_path=None):
        return self._read_data(path, adj_path)

    def _read_dev(self, path, adj_path=None):
        return self._read_data(path, adj_path)

    def _read_test(self, path, adj_path=None):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            line_dict = json.loads(line)
            example_id = line_dict["id"]
            candidate_character_list = [label_dict["label"] for label_dict in line_dict["question"]["choices"]]
            candidate_text_list = [label_dict["text"] for label_dict in line_dict["question"]["choices"]]
            candidate_num = len(candidate_text_list)
            context = [line_dict["question"]["stem"]] * candidate_num
            datable("example_id", example_id)
            datable("context", context)
            datable("candidate_text_list", candidate_text_list)
            self.label_vocab.add_sequence(candidate_character_list)
        return datable

    def read_all(self):
        return self._read_train(self.train_path, self.train_adj_path), \
               self._read_dev(self.dev_path, self.dev_adj_path), \
               self._read_test(self.test_path, self.test_adj_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}

    def read_addition(self):
        cp_emb = np.load(self.ent_emb_path)
        self.addition["cp_emb"] = cp_emb
        return self.addition


if __name__ == "__main__":
    reader = Commonsenseqa_Qagnn_Reader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/question_answering/CommonsenseQA_for_QAGNN/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    addition = reader.read_addition()
    print("end")
