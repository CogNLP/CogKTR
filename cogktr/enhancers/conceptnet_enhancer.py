from cogktr.enhancers.linker.conceptnet_linker import ConcetNetLinker
from cogktr.utils.general_utils import query_yes_no
from cogktr.utils.log_utils import logger
from cogktr.utils.io_utils import save_json,load_json
import os
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix

class ConceptNetEnhancer():
    def __init__(self, knowledge_graph_path, cache_path, reprocess=True):
        self.reprocess = reprocess
        self.conceptnet_linker = ConcetNetLinker(path=knowledge_graph_path)
        self.cache_path = os.path.abspath(cache_path)
        self.cached_file = os.path.join(self.cache_path,"conceptnet_enhanced_cache.json")

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

    def _enhance_qa(self, datable, statement, answer):
        if not os.path.exists(self.cache_path):
            if query_yes_no("Cache path {} does not exits.Do you want to create a new one?".format(self.cache_path),default="no"):
                os.makedirs(self.cache_path)
                logger.info("Created cached path {}.".format(self.cache_path))
            else:
                raise ValueError("Cached path {} is not valid!".format(self.cache_path))

        if not self.reprocess:
            enhanced_data_dict = load_json(self.cached_file)
        else:
            enhanced_data_dict = {}
            for statement, answer in tqdm(zip(datable[statement], datable[answer]),
                                                total=len(datable[statement])):
                all_concepts = self.conceptnet_linker.link(statement)
                answer_concepts = self.conceptnet_linker.link(answer)
                all_ids = set(self.conceptnet_linker.concept2id[c] for c in all_concepts)
                answer_ids = set(self.conceptnet_linker.concept2id[c] for c in answer_concepts)
                question_ids = all_ids - answer_ids

                qc_ids = sorted(question_ids)
                ac_ids = sorted(answer_ids)
                schema_graph = qc_ids + ac_ids
                arange = np.arange(len(schema_graph))
                qmask = arange < len(qc_ids)
                amask = (arange >= len(qc_ids)) & (arange < len(schema_graph))
                adj, concepts = self.concepts2adj(schema_graph)
                enhanced_data_dict.update({
                    statement:{
                        "adj":adj,
                        "concept":concepts,
                        "qmask":qmask,
                        "amask":amask,
                    }
                })
            save_json(enhanced_data_dict,self.cached_file)
        return enhanced_data_dict


    def enhance_train(self, datable, enhanced_key, enhanced_key_pair):
        return self._enhance_qa(
            datable=datable,
            statement=enhanced_key,
            answer=enhanced_key_pair,
        )


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

    enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key="statement",enhanced_key_pair="answer_text")

