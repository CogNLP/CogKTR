from cogktr.enhancers.linker.conceptnet_linker import ConcetNetLinker
from cogktr.utils.general_utils import query_yes_no
from cogktr.utils.log_utils import logger
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.io_utils import save_json, load_json, save_pickle, load_pickle
import os
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx


class ConceptNetEnhancer():
    def __init__(self,knowledge_graph_path, cache_path, reprocess=True,reprocess_conceptnet=False):
        self.reprocess = reprocess
        self.conceptnet_linker = ConcetNetLinker(path=knowledge_graph_path,reprocess=reprocess_conceptnet)
        self.cache_path = os.path.abspath(cache_path)
        self.meta_paths_set = set()

    def get_metapath_for_sample(self,adj, concept, qmask, amask):
        depth = 2
        n_relation = len(self.conceptnet_linker.relation2id)
        q_type = 'Q'
        a_type = "A"
        c_type = 'C'
        meta_path_for_sample = []
        G = nx.MultiDiGraph()
        assert len(concept) > 0
        adj = adj.toarray().reshape(n_relation, len(concept), len(concept))
        for i in range(len(concept)):
            G.add_node(i)
        triples = np.where(adj > 0)
        for r, s, t in zip(*triples):
            G.add_edge(s, t, rel=r)
        max_len = 0
        qnodes = np.where(qmask == True)[0]
        anodes = np.where(amask == True)[0]
        emask = ~(qmask | amask)
        for q in qnodes:
            for a in anodes:
                for edge_path in nx.all_simple_edge_paths(G, source=q, target=a, cutoff=depth):
                    meta_path = ""
                    pre = None
                    length = 0
                    for edge in edge_path:
                        s, t, key = edge
                        rel = G[s][t][key]['rel']
                        s_type = q_type if qmask[s] else c_type if emask[s] else a_type
                        t_type = q_type if qmask[t] else c_type if emask[t] else a_type
                        meta_path = meta_path + s_type + '-' + str(rel) + '-'
                        length += 1
                        if pre == None:
                            pre = t
                        else:
                            assert pre == s, (edge, pre, s)
                            pre = t
                    meta_path += t_type
                    if c_type not in meta_path:
                        max_len = max(max_len, length)
                        meta_path_for_sample.append(meta_path)
                for edge_path in nx.all_simple_edge_paths(G, source=a, target=q, cutoff=depth):
                    meta_path = ""
                    pre = None
                    length = 0
                    for edge in edge_path:
                        s, t, key = edge
                        rel = G[s][t][key]['rel']
                        s_type = q_type if qmask[s] else c_type if emask[s] else a_type
                        t_type = q_type if qmask[t] else c_type if emask[t] else a_type
                        meta_path = meta_path + s_type + '-' + str(rel) + '-'
                        length += 1
                        if pre == None:
                            pre = t
                        else:
                            assert pre == s, (edge, pre, s)
                            pre = t
                    meta_path += t_type
                    if c_type not in meta_path:
                        max_len = max(max_len, length)
                        meta_path_for_sample.append(meta_path)
        if len(meta_path_for_sample) == 0:
            length = 1
            max_len = max(max_len, length)
            no_qc_connect_rel_type = n_relation
            meta_path = "Q-" + str(no_qc_connect_rel_type) + "-A"
            meta_path_for_sample.append(meta_path)

        meta_path_for_sample_set = set(meta_path_for_sample)
        return (meta_path_for_sample, meta_path_for_sample_set)


    def concepts2adj(self, node_ids):
        id2relation = self.conceptnet_linker.id2relation
        cpnet = self.conceptnet_linker.conceptnet
        cids = np.array(node_ids, dtype=np.int32)
        n_rel = len(id2relation)
        n_node = cids.shape[0]
        adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
        for s in range(n_node):
            for t in range(n_node):
                s_c, t_c = cids[s], cids[t]
                if cpnet.has_edge(s_c, t_c):
                    for e_attr in cpnet[s_c][t_c].values():
                        if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                            adj[e_attr['rel']][s][t] = 1
        # cids += 1  # note!!! index 0 is reserved for padding
        try:
            adj = coo_matrix(adj.reshape(-1, n_node))
        except:
            adj = None
        return adj, cids

    def _enhance_qa(self, datable, statement, answer, cached_file):
        cached_file = os.path.join(self.cache_path, cached_file)
        if not os.path.exists(self.cache_path):
            if query_yes_no("Cache path {} does not exits.Do you want to create a new one?".format(self.cache_path),
                            default="no"):
                os.makedirs(self.cache_path)
                logger.info("Created cached path {}.".format(self.cache_path))
            else:
                raise ValueError("Cached path {} is not valid!".format(self.cache_path))

        if not self.reprocess:
            logger.info("Reading from cached file {}...".format(cached_file))
            enhanced_data_dict = load_pickle(cached_file)
        else:
            enhanced_data_dict = {}
            logger.info("Creating enhanced data and the cached file will be {}...".format(cached_file))
            for statement, answer in tqdm(zip(datable[statement], datable[answer]),
                                          total=len(datable[statement])):
                # link the concepts
                all_concepts = self.conceptnet_linker.link(statement)
                answer_concepts = self.conceptnet_linker.link(answer)
                all_ids = set(self.conceptnet_linker.concept2id[c] for c in all_concepts)
                answer_ids = set(self.conceptnet_linker.concept2id[c] for c in answer_concepts)
                question_ids = all_ids - answer_ids
                qc_ids = sorted(question_ids)
                ac_ids = sorted(answer_ids)

                # extract subgraph
                schema_graph = qc_ids + ac_ids
                arange = np.arange(len(schema_graph))
                qmask = arange < len(qc_ids)
                amask = (arange >= len(qc_ids)) & (arange < len(schema_graph))
                adj, concepts = self.concepts2adj(schema_graph)

                # compute meta path in the subgraph
                meta_paths_list,meta_paths_set = self.get_metapath_for_sample(adj,concepts,qmask, amask)
                self.meta_paths_set.update(meta_paths_set)
                # update enhanced data dict
                enhanced_data_dict.update({
                    statement: {
                        "adj": adj,
                        "concept": concepts,
                        "qmask": qmask,
                        "amask": amask,
                        "meta_paths_list":meta_paths_list,
                        "meta_paths_set":meta_paths_set,
                    }
                })
            save_pickle(enhanced_data_dict, cached_file)
            logger.info("Successfully created the cached file {}!".format(cached_file))
        return enhanced_data_dict

    def enhance_train(self, datable, enhanced_key, enhanced_key_pair):
        return self._enhance_qa(
            datable=datable,
            statement=enhanced_key,
            answer=enhanced_key_pair,
            cached_file="conceptnet_enhanced_cache_train.pkl"
        )

    def enhance_dev(self, datable, enhanced_key, enhanced_key_pair):
        return self._enhance_qa(
            datable=datable,
            statement=enhanced_key,
            answer=enhanced_key_pair,
            cached_file="conceptnet_enhanced_cache_dev.pkl"
        )

    def enhance_test(self, datable, enhanced_key, enhanced_key_pair):
        return self._enhance_qa(
            datable=datable,
            statement=enhanced_key,
            answer=enhanced_key_pair,
            cached_file="conceptnet_enhanced_cache_test.pkl"
        )

    def enhance_all(self,train_data,dev_data,test_data,vocab=None):
        enhanced_train_dict = self.enhance_train(train_data, enhanced_key="statement",
                                                     enhanced_key_pair="answer_text")
        enhanced_dev_dict = self.enhance_dev(dev_data, enhanced_key="statement", enhanced_key_pair="answer_text")
        enhanced_test_dict = self.enhance_test(test_data, enhanced_key="statement",
                                                   enhanced_key_pair="answer_text")
        if isinstance(vocab,dict):
            if self.reprocess:
                meta_paths_set =self.meta_paths_set
            else:
                meta_paths_set = set()
                for data_dict in [enhanced_train_dict,enhanced_dev_dict,enhanced_test_dict]:
                    for key,value in data_dict.items():
                        meta_paths_set.update(value["meta_paths_set"])


            cpnet_vocab = Vocabulary()
            cpnet_vocab.add_sequence(list(meta_paths_set))
            cpnet_vocab.create()
            vocab["metapath"] = cpnet_vocab
            self.meta_paths_set = set()

        return enhanced_train_dict,enhanced_dev_dict,enhanced_test_dict


if __name__ == '__main__':
    enhancer = ConceptNetEnhancer(
        knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet",
        cache_path='/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/enhanced_data',
        reprocess=True
    )

    from cogktr import OpenBookQAReader

    reader = OpenBookQAReader(
        raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    # enhanced_train_dict = enhancer.enhance_train(train_data, enhanced_key="statement", enhanced_key_pair="answer_text")
    # enhanced_dev_dict = enhancer.enhance_dev(dev_data, enhanced_key="statement", enhanced_key_pair="answer_text")
    # enhanced_test_dict = enhancer.enhance_test(test_data, enhanced_key="statement", enhanced_key_pair="answer_text")
    # save_pickle(enhancer.meta_paths_set,'/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/enhanced_data/meta_path_set.pkl')

