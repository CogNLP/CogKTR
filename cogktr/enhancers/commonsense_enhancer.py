import os
from cogktr.enhancers.base_enhancer import BaseEnhancer
from nltk.tokenize import word_tokenize
from cogktr.utils.io_utils import save_json, load_json
from tqdm import tqdm
from cogktr.enhancers.linker.conceptnet_linker import ConcetNetLinker
from cogktr.utils.general_utils import query_yes_no, create_cache
from cogktr.utils.log_utils import logger
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.io_utils import save_json, load_json, save_pickle, load_pickle
import os
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx


class CommonsenseEnhancer(BaseEnhancer):
    def __init__(self, knowledge_graph_path, cache_path, cache_file, reprocess=False, load_conceptnet=True):
        super(CommonsenseEnhancer, self).__init__()
        self.reprocess = reprocess
        self.load_conceptnet = load_conceptnet
        self.cache_path = os.path.abspath(cache_path)
        self.cache_file = cache_file
        create_cache(self.cache_path, self.cache_file)
        self.cache_path_file = os.path.join(cache_path, cache_file)

        if self.load_conceptnet:
            self.conceptnet_linker = ConcetNetLinker(path=knowledge_graph_path)
            self.meta_paths_set = set()

    def enhance_sentence(self, sentence):
        enhanced_dict = {}

        if isinstance(sentence, list):
            words = sentence
            sentence = tuple(sentence)
        elif isinstance(sentence, str):
            words = word_tokenize(sentence)
        else:
            ValueError("Sentence must be str or a list of words!")
        enhanced_dict[sentence] = {}
        enhanced_dict[sentence]["words"] = words

        result = self.conceptnet_linker.link(words)
        enhanced_dict[sentence]["knowledge"] = result["knowledge"]

        # concepts = self.conceptnet_linker.link(words)
        # enhanced_dict[sentence]["concepts"] = concepts

        return enhanced_dict

    def enhance_sentence_pair(self, sentence, sentence_pair, return_metapath=True):
        enhanced_dict = {}
        enhanced_sentence_dict = self.enhance_sentence(sentence)
        enhanced_sentence_pair_dict = self.enhance_sentence(sentence_pair)
        sentence_key = list(enhanced_sentence_dict.keys())[0]
        sentence_pair_key = list(enhanced_sentence_pair_dict.keys())[0]
        enhanced_dict[(sentence_key, sentence_pair_key)] = {}
        enhanced_dict[(sentence_key, sentence_pair_key)]["sentence"] = enhanced_sentence_dict[sentence_key]
        enhanced_dict[(sentence_key, sentence_pair_key)]["sentence_pair"] = enhanced_sentence_pair_dict[sentence_pair_key]
        enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"] = {}
        if return_metapath:
            f = lambda x:[concept for sample in x for concept in sample["concepts"]]
            all_concepts = f(enhanced_sentence_dict[sentence]["knowledge"])
            answer_concepts = f(enhanced_sentence_pair_dict[sentence_pair]["knowledge"])
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
            meta_paths_list, meta_paths_set = self.get_metapath_for_sample(adj, concepts, qmask, amask)
            self.meta_paths_set.update(meta_paths_set)
            # update enhanced data dict
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["adj"] = adj
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["concept"] = concepts
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["qmask"] = qmask
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["amask"] = amask
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["meta_paths_list"] = meta_paths_list
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["meta_paths_set"] = meta_paths_set

        return enhanced_dict

    def _enhance_data(self,datable,dict_name=None,enhanced_key_1=None,enhanced_key_2=None,return_metapath=False):
        enhanced_dict = {}
        if not self.reprocess and os.path.exists(os.path.join(self.cache_path_file, dict_name)):
            enhanced_dict = load_pickle(os.path.join(self.cache_path_file, dict_name))
        else:
            logger.info("Enhancing data...")
            if enhanced_key_2 is None:
                for sentence in tqdm(datable[enhanced_key_1]):
                    enhanced_sentence_dict = self.enhance_sentence(sentence=sentence,)
                    enhanced_dict.update(enhanced_sentence_dict)
                save_pickle(enhanced_dict, os.path.join(self.cache_path_file, dict_name))
            else:
                for sentence, sentence_pair in tqdm(zip(datable[enhanced_key_1], datable[enhanced_key_2]),
                                                    total=len(datable[enhanced_key_1])):
                    enhanced_sentence_pair = self.enhance_sentence_pair(sentence=sentence,
                                                                        sentence_pair=sentence_pair,
                                                                        return_metapath=return_metapath)
                    enhanced_dict.update(enhanced_sentence_pair)
                save_pickle(enhanced_dict, os.path.join(self.cache_path_file, dict_name))
        return enhanced_dict

    def enhance_train(self,
                      datable,
                      enhanced_key_1="sentence",
                      enhanced_key_2=None,
                      return_metapath=False,
                      ):
        tags = []
        if return_metapath:
            tags.append("metapath")
        if len(tags) == 0:
            dict_name = "enhanced_train"
        else:
            dict_name = "enhanced_train_" + "_".join(tags) + ".pkl"
        return self._enhance_data(datable=datable,
                                  dict_name=dict_name,
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_metapath=return_metapath)

    def enhance_dev(self,
                    datable,
                    enhanced_key_1="sentence",
                    enhanced_key_2=None,
                    return_metapath=False,
                    ):
        tags = []
        if return_metapath:
            tags.append("metapath")
        if len(tags) == 0:
            dict_name = "enhanced_dev"
        else:
            dict_name = "enhanced_dev_" + "_".join(tags) + ".pkl"
        return self._enhance_data(datable=datable,
                                  dict_name=dict_name,
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_metapath=return_metapath)

    def enhance_test(self,
                     datable,
                     enhanced_key_1="sentence",
                     enhanced_key_2=None,
                     return_metapath=False,
                     ):
        tags = []
        if return_metapath:
            tags.append("metapath")
        if len(tags) == 0:
            dict_name = "enhanced_test"
        else:
            dict_name = "enhanced_test_" + "_".join(tags) + ".pkl"
        return self._enhance_data(datable=datable,
                                  dict_name=dict_name,
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_metapath=return_metapath)

    def enhance_all(self,train_data,dev_data,test_data,vocab=None,return_metapath=False):
        enhanced_train_dict = self.enhance_train(train_data, enhanced_key_1="statement",enhanced_key_2="answer_text",return_metapath=return_metapath)
        enhanced_dev_dict = self.enhance_dev(dev_data, enhanced_key_1="statement", enhanced_key_2="answer_text",return_metapath=return_metapath)
        enhanced_test_dict = self.enhance_test(test_data, enhanced_key_1="statement",enhanced_key_2="answer_text",return_metapath=return_metapath)

        if isinstance(vocab,dict):
            if self.reprocess:
                meta_paths_set =self.meta_paths_set
            else:
                meta_paths_set = set()
                for data_dict in [enhanced_train_dict,enhanced_dev_dict,enhanced_test_dict]:
                    for key,value in data_dict.items():
                        meta_paths_set.update(value["interaction"]["meta_paths_set"] )

            cpnet_vocab = Vocabulary()
            cpnet_vocab.add_sequence(list(meta_paths_set))
            cpnet_vocab.create()
            vocab["metapath"] = cpnet_vocab
            self.meta_paths_set = set()

        return enhanced_train_dict,enhanced_dev_dict,enhanced_test_dict

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

if __name__ == '__main__':
    enhancer = CommonsenseEnhancer(
        knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet",
        cache_path='/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/',
        cache_file="enhanced_data",
        reprocess=True,
        load_conceptnet=True
    )
    from cogktr import OpenBookQAReader
    reader = OpenBookQAReader(
        raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    enhanced_train_dict,enhanced_dev_dict,enhanced_test_dict = enhancer.enhance_all(train_data,dev_data,test_data,vocab,return_metapath=True)



