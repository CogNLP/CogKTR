import numpy as np

from cogktr.enhancers.embedder import BaseEmbedder
import os
from cogktr.utils.constant.conceptnet_constants.constants import *
from cogktr.utils.log_utils import logger
from tqdm import tqdm
import json
import os
import networkx as nx
import spacy

class ConceptnetEmbedder(BaseEmbedder):
    def __init__(self, path):
        super(ConceptnetEmbedder, self).__init__()
        if not os.path.exists(path):
            raise ValueError("Path {} does not exits!".format(path))
        self.pattern_path = os.path.join(path, "matcher_patterns.json")
        self.vocab_path = os.path.join(path, "concept.txt")
        self.conceptnet_path = os.path.join(path,"conceptnet-assertions-5.6.0.csv")
        self.concetnet_en_path = os.path.join(path,"conceptnet.en.csv")
        self.concet_embedding_path = os.path.join(path, "tzw.ent.npy")
        if not os.path.exists(self.conceptnet_path):
            raise FileNotFoundError("Original conceptnet file {} not found!".format(self.conceptnet_path))
        if not os.path.exists(self.concet_embedding_path):
            os.system("wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy -P {}".format(os.path.abspath(path)))
        self.embedding = np.load(self.concet_embedding_path)
        self.id2relation = MERGED_RELATIONS
        self.relation2id = {r: i for i, r in enumerate(self.id2relation)}

        if not os.path.exists(self.concetnet_en_path) or not os.path.exists(self.vocab_path):
            extract_english(
                conceptnet_path=self.conceptnet_path,
                output_csv_path=self.concetnet_en_path,
                output_vocab_path=self.vocab_path,
            )

        with open(self.vocab_path, "r", encoding="utf8") as fin:
            self.id2concept = [w.strip() for w in fin]
        self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
        self.vocab = {
            "id2relation": self.id2relation,
            "relation2id": self.relation2id,
            "id2concept": self.id2concept,
            "concept2id": self.concept2id,
        }

    def embed(self,concept):
        if concept not in self.concept2id:
            raise ValueError("{} not found in conceptnet!".format(concept))
        return self.embedding[self.concept2id[concept]]


def extract_english(conceptnet_path, output_csv_path, output_vocab_path):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    logger.info('extracting English concepts and relations from ConceptNet...')
    relation_mapping = load_merge_relation()
    num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
    cpnet_vocab = []
    concepts_seen = set()
    with open(conceptnet_path, 'r', encoding="utf8") as fin, \
            open(output_csv_path, 'w', encoding="utf8") as fout:
        for line in tqdm(fin, total=num_lines):
            toks = line.strip().split('\t')
            if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = toks[1].split("/")[-1].lower()
                head = del_pos(toks[2]).split("/")[-1].lower()
                tail = del_pos(toks[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if rel not in relation_mapping:
                    continue

                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    head, tail, rel = tail, head, rel[1:]

                data = json.loads(toks[4])

                fout.write('\t'.join([rel, head, tail, str(data["weight"])]) + '\n')

                for w in [head, tail]:
                    if w not in concepts_seen:
                        concepts_seen.add(w)
                        cpnet_vocab.append(w)

    with open(output_vocab_path, 'w') as fout:
        for word in cpnet_vocab:
            fout.write(word + '\n')

    logger.info(f'extracted ConceptNet csv file saved to {output_csv_path}')
    logger.info(f'extracted concept vocabulary saved to {output_vocab_path}')

def load_merge_relation():
    relation_mapping = dict()
    for line in RELATION_GROUPS:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s

if __name__ == "__main__":
    embedder = ConceptnetEmbedder(path='/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet/',)
    result = embedder.embed("bert")