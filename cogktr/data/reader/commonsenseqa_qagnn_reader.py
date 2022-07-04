import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import numpy as np
import json
import pickle


class CommonsenseqaQagnnReader(BaseReader):
    def __init__(self, raw_data_path, use_cache=True):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.use_cache = use_cache
        downloader = Downloader()
        downloader.download_commonsenseqa_qagnn_raw_data(raw_data_path)
        self.train_file = 'statement/train.statement.jsonl'
        self.dev_file = 'statement/dev.statement.jsonl'
        self.test_file = 'statement/test.statement.jsonl'
        self.train_adj_file = 'graph/train.graph.adj.pk'
        self.dev_adj_file = 'graph/dev.graph.adj.pk'
        self.test_adj_file = 'graph/test.graph.adj.pk'
        self.train_adj_cache_file = 'graph/train.graph.adj.pk.loaded_cache'
        self.dev_adj_cache_file = 'graph/dev.graph.adj.pk.loaded_cache'
        self.test_adj_cache_file = 'graph/test.graph.adj.pk.loaded_cache'
        self.ent_emb_file = 'cpnet/tzw.ent.npy'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.train_adj_path = os.path.join(raw_data_path, self.train_adj_file)
        self.dev_adj_path = os.path.join(raw_data_path, self.dev_adj_file)
        self.test_adj_path = os.path.join(raw_data_path, self.test_adj_file)
        self.train_adj_cache_path = os.path.join(raw_data_path, self.train_adj_cache_file)
        self.dev_adj_cache_path = os.path.join(raw_data_path, self.dev_adj_cache_file)
        self.test_adj_cache_path = os.path.join(raw_data_path, self.test_adj_cache_file)
        self.ent_emb_path = os.path.join(raw_data_path, self.ent_emb_file)
        self.label_num = 5
        self.label_vocab = Vocabulary()
        self.addition = {}

    def _read_data(self, path, adj_path=None, adj_cache_path=None):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        with open(adj_path, 'rb') as adj_file:
            adj_concept_pairs = pickle.load(adj_file)
        for line, adj_concept_pair in zip(lines, adj_concept_pairs):
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
            if not self.use_cache:
                adj = adj_concept_pair["adj"]
                concepts = adj_concept_pair["concepts"]
                qmask = adj_concept_pair["qmask"]
                amask = adj_concept_pair["amask"]
                cid2score = adj_concept_pair["cid2score"]
                datable("adj", adj)
                datable("concepts", concepts)
                datable("qmask", qmask)
                datable("amask", amask)
                datable("cid2score", cid2score)
            self.label_vocab.add_sequence(candidate_character_list)

        if self.use_cache:
            adj_cache = self._read_cache(adj_cache_path)
            adj_lengths_oris = list(adj_cache["adj_lengths_ori"])
            concept_ids = list(adj_cache["concept_ids"])
            node_type_ids = list(adj_cache["node_type_ids"])
            node_scores = list(adj_cache["node_scores"])
            adj_lengths = list(adj_cache["adj_lengths"])
            edge_indexes = adj_cache["edge_index"]
            edge_types = adj_cache["edge_type"]
            adj_lengths_ori_list = []
            concept_id_list = []
            node_type_id_list = []
            node_score_list = []
            adj_length_list = []
            edge_index_list = []
            edge_type_list = []
            for adj_lengths_ori, concept_id, node_type_id, node_score, adj_length, edge_index, edge_type in zip(
                    adj_lengths_oris, concept_ids, node_type_ids, node_scores, adj_lengths, edge_indexes, edge_types):
                adj_lengths_ori_list.append(adj_lengths_ori)
                concept_id_list.append(concept_id)
                node_type_id_list.append(node_type_id)
                node_score_list.append(node_score)
                adj_length_list.append(adj_length)
                edge_index_list.append(edge_index)
                edge_type_list.append(edge_type)
                if len(concept_id_list) == 5:
                    datable("adj_lengths_ori", adj_lengths_ori_list)
                    datable("concept_id", concept_id_list)
                    datable("node_type_id", node_type_id_list)
                    datable("node_score", node_score_list)
                    datable("adj_length", adj_length_list)
                    datable("edge_index", edge_index_list)
                    datable("edge_type", edge_type_list)
                    adj_lengths_ori_list = []
                    concept_id_list = []
                    node_type_id_list = []
                    node_score_list = []
                    adj_length_list = []
                    edge_index_list = []
                    edge_type_list = []
        return datable

    def _read_train(self, path, adj_path=None, adj_cache_path=None):
        return self._read_data(path, adj_path, adj_cache_path)

    def _read_dev(self, path, adj_path=None, adj_cache_path=None):
        return self._read_data(path, adj_path, adj_cache_path)

    def _read_test(self, path, adj_path=None, adj_cache_path=None):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        with open(adj_path, 'rb') as adj_file:
            adj_concept_pairs = pickle.load(adj_file)
        for line, adj_concept_pair in zip(lines, adj_concept_pairs):
            line_dict = json.loads(line)
            example_id = line_dict["id"]
            candidate_character_list = [label_dict["label"] for label_dict in line_dict["question"]["choices"]]
            candidate_text_list = [label_dict["text"] for label_dict in line_dict["question"]["choices"]]
            candidate_num = len(candidate_text_list)
            context = [line_dict["question"]["stem"]] * candidate_num
            datable("example_id", example_id)
            datable("context", context)
            datable("candidate_text_list", candidate_text_list)
            if not self.use_cache:
                adj = adj_concept_pair["adj"]
                concepts = adj_concept_pair["concepts"]
                qmask = adj_concept_pair["qmask"]
                amask = adj_concept_pair["amask"]
                cid2score = adj_concept_pair["cid2score"]
                datable("adj", adj)
                datable("concepts", concepts)
                datable("qmask", qmask)
                datable("amask", amask)
                datable("cid2score", cid2score)
            self.label_vocab.add_sequence(candidate_character_list)

        if self.use_cache:
            adj_cache = self._read_cache(adj_cache_path)
            adj_lengths_oris = list(adj_cache["adj_lengths_ori"])
            concept_ids = list(adj_cache["concept_ids"])
            node_type_ids = list(adj_cache["node_type_ids"])
            node_scores = list(adj_cache["node_scores"])
            adj_lengths = list(adj_cache["adj_lengths"])
            edge_indexes = adj_cache["edge_index"]
            edge_types = adj_cache["edge_type"]
            adj_lengths_ori_list = []
            concept_id_list = []
            node_type_id_list = []
            node_score_list = []
            adj_length_list = []
            edge_index_list = []
            edge_type_list = []
            for adj_lengths_ori, concept_id, node_type_id, node_score, adj_length, edge_index, edge_type in zip(
                    adj_lengths_oris, concept_ids, node_type_ids, node_scores, adj_lengths, edge_indexes, edge_types):
                adj_lengths_ori_list.append(adj_lengths_ori)
                concept_id_list.append(concept_id)
                node_type_id_list.append(node_type_id)
                node_score_list.append(node_score)
                adj_length_list.append(adj_length)
                edge_index_list.append(edge_index)
                edge_type_list.append(edge_type)
                if len(concept_id_list) == 5:
                    datable("adj_lengths_ori", adj_lengths_ori_list)
                    datable("concept_id", concept_id_list)
                    datable("node_type_id", node_type_id_list)
                    datable("node_score", node_score_list)
                    datable("adj_length", adj_length_list)
                    datable("edge_index", edge_index_list)
                    datable("edge_type", edge_type_list)
                    adj_lengths_ori_list = []
                    concept_id_list = []
                    node_type_id_list = []
                    node_score_list = []
                    adj_length_list = []
                    edge_index_list = []
                    edge_type_list = []
        return datable

    def read_all(self):
        return self._read_train(self.train_path, self.train_adj_path, self.train_adj_cache_path), \
               self._read_dev(self.dev_path, self.dev_adj_path, self.dev_adj_cache_path), \
               self._read_test(self.test_path, self.test_adj_path, self.test_adj_cache_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}

    def _read_cache(self, path):
        with open(path, 'rb') as file:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel = pickle.load(
                file)
            adj_cache = {}
            adj_cache["adj_lengths_ori"] = adj_lengths_ori
            adj_cache["concept_ids"] = concept_ids
            adj_cache["node_type_ids"] = node_type_ids
            adj_cache["node_scores"] = node_scores
            adj_cache["adj_lengths"] = adj_lengths
            adj_cache["edge_index"] = edge_index
            adj_cache["edge_type"] = edge_type
            adj_cache["half_n_rel"] = half_n_rel
        return adj_cache

    def read_addition(self):
        cp_emb = np.load(self.ent_emb_path)
        self.addition["cp_emb"] = cp_emb
        self.addition["use_cache"] = self.use_cache
        return self.addition


if __name__ == "__main__":
    reader = CommonsenseqaQagnnReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/question_answering/CommonsenseQA_for_QAGNN/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    addition = reader.read_addition()
    print("end")
