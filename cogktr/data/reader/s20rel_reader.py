from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
import os


class S20relReader(BaseReader):
    def __init__(self, data_dir, n_partition=20, name="Node Prediction With Partition",):
        super().__init__()
        self.NAME = name
        self.id2ent = {}
        self.id2rel = {}
        self.n_partition = n_partition
        self.label_vocab = Vocabulary()
        self.tri_file = os.path.join(data_dir, "train2id.txt")
        self.ent_file = os.path.join(data_dir, "entity2id.txt")
        self.rel_file = os.path.join(data_dir, "relation2id.txt")
        self.partition_file = os.path.join(data_dir, "partition_20.txt")

        # get id2ent
        with open(self.ent_file, "r") as f:
            self.ent_total = (int)(f.readline())
            for ent in f.readlines():
                self.id2ent[int(ent.split("\t")[1].strip())] = ent.split("\t")[0]
            print(
                f"Loading entities ent_total:{self.ent_total} len(self.id2ent): {len(self.id2ent)}"
            )

        # get rel2id
        with open(self.rel_file, "r") as f:
            print("Read Relation File")
            self.rel_total = (int)(f.readline())
            for rel in f.readlines():
                self.id2rel[int(rel.split("\t")[1].strip())] = rel.split("\t")[0]
            print(f"{len(self.id2rel)} relations loaded.")

        # Read Partition File
        self.node_group_idx = {}  # A dict saving the group index of each entity
        self.num_class_list = []  # A list saving the number of entities for each group
        self.nodes_partition = []  # A list of node dicts

        with open(self.partition_file, "r") as f:
            print(f"Reading partition file: {self.partition_file}.")
            for group_idx, line in enumerate(f.readlines()):
                nodes = {
                    int(eid.strip()): idx for idx, eid in enumerate(line.split("\t"))
                }
                for eid in line.split("\t"):
                    self.node_group_idx[int(eid.strip())] = group_idx
                self.nodes_partition.append(nodes)
                self.num_class_list.append(len(nodes))

            print(f"{len(self.nodes_partition)} partitioned groups loaded. ")
            print(
                f"Number for nodes in each partitions: min({min(self.num_class_list)}),max({max(self.num_class_list)})"
            )
            print(
                f"Total Nodes number: {self.ent_total}, Nodes number in partitions:{len(self.node_group_idx)}"
            )

        # Read Triple File
        f = open(self.tri_file, "r")
        triples_total = (int)(f.readline())

        count = 0
        self.triple_list = [[] for i in range(self.n_partition)]
        for line in f.readlines():
            h, t, r = line.strip().split("\t")
            if (
                    ((int)(h) in self.id2ent)
                    and ((int)(t) in self.id2ent)
                    and ((int)(r) in self.id2rel)
                    and ((int)(h) in self.node_group_idx)
                    and ((int)(t) in self.node_group_idx)
            ):
                group_idx_h = self.node_group_idx[(int)(h)]
                group_idx_t = self.node_group_idx[(int)(t)]
                if group_idx_h == group_idx_t:
                    self.triple_list[group_idx_t].append(((int)(h), (int)(t), (int)(r)))
                    count += 1
        f.close()
        if triples_total != count:
            print(
                f"Using sub-set mode or some triples are missing, total:{triples_total} --> subset:{count}"
            )
            # triples_total = count

    def _read_data(self, group_idx: int):
        datable = DataTable()

        def cls_one_hot(ent_id):
            return self.nodes_partition[group_idx][ent_id]

        for (h_id, t_id, r_id) in self.triple_list[group_idx]:
            text_h = self.id2ent[h_id]
            text_t = self.id2ent[t_id]
            text_r = self.id2rel[r_id]
            label=cls_one_hot(t_id)
            self.label_vocab.add(label)

            datable("text_head", text_h)
            datable("text_relation", text_r)
            datable("label", label)
        print(
            f"Get {len(datable)} examples of {self.NAME} datasets from partition {group_idx}/{self.n_partition} set"
        )
        return datable

    def read_train(self, group_idx: int):
        return self._read_data(group_idx)

    def read_vocab(self):
        self.label_vocab.create()
        # return {"label_vocab": self.label_vocab}
        return {}


if __name__ == "__main__":
    s20mop_reader = S20relReader("/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/S20Rel")
    train_dataset = s20mop_reader.read_train(group_idx=0)
    vocab = s20mop_reader.read_vocab()