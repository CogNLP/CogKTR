from cogktr.enhancers.searcher import BaseSearcher
from cogktr.utils.constant.conceptnet_constants.constants import *
from tqdm import tqdm
import json
import os
import networkx as nx
import spacy
import nltk
from cogktr.utils.log_utils import logger
from spacy.matcher import Matcher
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.io_utils import save_pickle,load_pickle


class ConcetNetSearcher(BaseSearcher):
    def __init__(self, path):
        super(ConcetNetSearcher, self).__init__()
        if not os.path.exists(path):
            raise ValueError("Path {} does not exits!".format(path))
        self.vocab_path = os.path.join(path, "concept.txt")
        self.conceptnet_path = os.path.join(path,"conceptnet-assertions-5.6.0.csv")
        self.concetnet_en_path = os.path.join(path,"conceptnet.en.csv")
        self.pruned_graph_path = os.path.join(path,"conceptnet.en.pruned.graph")
        self.unpruned_graph_path = os.path.join(path,"conceptnet.en.unpruned.graph")
        if not os.path.exists(self.conceptnet_path):
            raise FileNotFoundError("Original conceptnet file {} not found!".format(self.conceptnet_path))

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

        if not os.path.exists(self.unpruned_graph_path):
            construct_graph(
                cpnet_csv_path=self.concetnet_en_path,
                cpnet_vocab=self.vocab,
                output_path=self.unpruned_graph_path,
                prune=False,
            )
        if not os.path.exists(self.pruned_graph_path):
            construct_graph(
                cpnet_csv_path=self.concetnet_en_path,
                cpnet_vocab=self.vocab,
                output_path=self.pruned_graph_path,
                prune=True,
            )
        self.conceptnet = nx.read_gpickle(os.path.join(path, "conceptnet.en.pruned.graph"))
        conceptnet_simple = nx.Graph()
        for u, v, data in self.conceptnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if conceptnet_simple.has_edge(u, v):
                conceptnet_simple[u][v]['weight'] += w
            else:
                conceptnet_simple.add_edge(u, v, weight=w)
        self.conceptnet_simple = conceptnet_simple

    def search(self, concepts,multi_hop=2):
        concept_ids = [self.concept2id[concept] for concept in concepts]

    def search_between_two_nodes(self,concept_a,concept_b,multi_hop):
        if not isinstance(concept_a,str) and not isinstance(concept_a,int):
            raise ValueError("concept_a got type {}!".format(type(concept_a)))
        if not isinstance(concept_b,str) and not isinstance(concept_b,int):
            raise ValueError("concept_b got type {}!".format(type(concept_b)))
        concept_a_id = concept_a if isinstance(concept_a,int) else self.concept2id[concept_a]
        concept_b_id = concept_b if isinstance(concept_b, int) else self.concept2id[concept_b]
        path_generator = nx.all_simple_edge_paths(self.conceptnet,concept_a_id,concept_b_id,cutoff=multi_hop)
        results = []
        n_relation = len(self.id2relation)
        for paths in path_generator:
            triples = []
            for source,destination,_ in paths:
                for e_attr in self.conceptnet[source][destination].values():
                    relation_id = e_attr['rel']
                    if relation_id >= 0 and relation_id < n_relation:
                        relation = self.id2relation[relation_id]
                        triples.append([self.id2concept[source], relation, self.id2concept[destination]])
                    else:
                        relation = self.id2relation[relation_id - n_relation]
                        triples.append([self.id2concept[destination],relation,self.id2concept[source]])
            results.append(triples)
        return results

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


def construct_graph(cpnet_csv_path, cpnet_vocab, output_path, prune=True):
    logger.info("Generating ConceptNet graph files with prune={}".format(str(prune)))

    # id2concept = cpnet_vocab["id2conceptnet"]
    id2relation = cpnet_vocab["id2relation"]
    concept2id = cpnet_vocab["concept2id"]
    relation2id = cpnet_vocab["relation2id"]

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:

        def not_save(cpt):
            if cpt in BLACKLIST:
                return True
            return False

        attrs = set()

        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            if subj == obj:  # delete loops
                continue
            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                attrs.add((obj, subj, rel + len(relation2id)))

    nx.write_gpickle(graph, output_path)
    logger.info(f"graph file saved to {output_path}")

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



if __name__ == '__main__':
    print("Hello World!")
    conceptnet_searcher = ConcetNetSearcher(path='/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet/',)
    results = conceptnet_searcher.search_between_two_nodes("bert","sesame_street",multi_hop=4)
    for paths in results:
        print(paths)




