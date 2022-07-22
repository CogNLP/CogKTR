import os
import time

from cogktr.enhancers.base_enhancer import BaseEnhancer
from cogktr.enhancers.linker.wikipedia_linker import WikipediaLinker
from cogktr.enhancers.embedder.wikipedia_embedder import WikipediaEmbedder
from cogktr.enhancers.searcher.wikipedia_searcher import WikipediaSearcher
from cogktr.enhancers.searcher.wikidata_searcher import WikidataSearcher
from cogktr.utils.io_utils import save_pickle, load_pickle
from tqdm import tqdm


class WorldEnhancer(BaseEnhancer):
    def __init__(self,
                 knowledge_graph_path,
                 cache_path,
                 cache_file,
                 reprocess,
                 load_entity_desc,
                 load_entity_embedding,
                 load_entity_kg):
        super().__init__()
        if not os.path.exists(cache_path):
            raise FileExistsError("{} PATH does not exist!".format(cache_path))
        if not os.path.exists(os.path.join(cache_path, cache_file)):
            os.makedirs(os.path.join(cache_path, cache_file))

        self.knowledge_graph_path = knowledge_graph_path
        self.cache_path = cache_path
        self.cache_file = cache_file
        self.cache_path_file = os.path.join(cache_path, cache_file)
        self.reprocess = reprocess

        self.root_path = knowledge_graph_path

        if self.reprocess:
            if load_entity_desc or load_entity_embedding or load_entity_kg:
                self.wikipedia_linker_tool = "cogie"
                self.wikipedia_linker = WikipediaLinker(tool=self.wikipedia_linker_tool)

            if load_entity_desc:
                self.wikipedia_searcher_tool = "blink"
                self.wikipedia_searcher_path = os.path.join(self.root_path, "wikipedia/entity.jsonl")
                self.wikipedia_searcher = WikipediaSearcher(tool=self.wikipedia_searcher_tool,
                                                            path=self.wikipedia_searcher_path)

            if load_entity_embedding:
                self.wikipedia_embedder_tool = "wikipedia2vec"
                self.wikipedia_embedder_path = os.path.join(self.root_path,
                                                            "wikipedia2vec/enwiki_20180420_win10_100d.pkl")
                self.wikipedia_embedder = WikipediaEmbedder(tool=self.wikipedia_embedder_tool,
                                                            path=self.wikipedia_embedder_path)

            if load_entity_kg:
                self.wikidata_searcher = WikidataSearcher()

    def enhance_sentence(self, sentence,
                         return_entity_desc=False,
                         return_entity_embedding=False,
                         return_entity_kg=False):

        enhanced_dict = {}
        link_dict = self.wikipedia_linker.link(sentence)

        if isinstance(sentence, list):
            sentence = tuple(sentence)

        enhanced_dict[sentence] = {}
        enhanced_dict[sentence]["words"] = link_dict["words"]
        enhanced_dict[sentence]["entities"] = link_dict["spans"]

        if return_entity_desc:
            for entity in enhanced_dict[sentence]["entities"]:
                entity["desc"] = self.wikipedia_searcher.search(entity["wikipedia_id"])["desc"]

        if return_entity_embedding:
            for entity in enhanced_dict[sentence]["entities"]:
                entity["entity_embedding"] = self.wikipedia_embedder.embed(entity["entity_title"])["entity_embedding"]

        if return_entity_kg:
            for entity in enhanced_dict[sentence]["entities"]:
                entity["kg"] = self.wikidata_searcher.search(wikipedia_id=entity["wikipedia_id"],
                                                             result_num=10)
            time.sleep(0.5)

        return enhanced_dict

    def enhance_sentence_pair(self, sentence, sentence_pair,
                              return_entity_desc=False,
                              return_entity_embedding=False,
                              return_entity_kg=False):
        enhanced_dict = {}
        enhanced_sentence_dict = self.enhance_sentence(sentence=sentence,
                                                       return_entity_desc=return_entity_desc,
                                                       return_entity_embedding=return_entity_embedding,
                                                       return_entity_kg=return_entity_kg)
        enhanced_sentence_pair_dict = self.enhance_sentence(sentence=sentence_pair,
                                                            return_entity_desc=return_entity_desc,
                                                            return_entity_embedding=return_entity_embedding,
                                                            return_entity_kg=return_entity_kg)
        sentence_key = list(enhanced_sentence_dict.keys())[0]
        sentence_pair_key = list(enhanced_sentence_pair_dict.keys())[0]
        enhanced_dict[(sentence_key, sentence_pair_key)] = {}
        enhanced_dict[(sentence_key, sentence_pair_key)]["sentence"] = enhanced_sentence_dict[sentence_key]
        enhanced_dict[(sentence_key, sentence_pair_key)]["sentence_pair"] = enhanced_sentence_pair_dict[
            sentence_pair_key]
        enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"] = {}

        return enhanced_dict

    def _enhance_data(self,
                      datable,
                      dict_name=None,
                      enhanced_key_1=None,
                      enhanced_key_2=None,
                      return_entity_desc=False,
                      return_entity_embedding=False,
                      return_entity_kg=False):
        enhanced_dict = {}
        if not self.reprocess and os.path.exists(os.path.join(self.cache_path_file, dict_name)):
            enhanced_dict = load_pickle(os.path.join(self.cache_path_file, dict_name))
        else:
            print("Enhancing data...")
            if enhanced_key_2 is None:
                for sentence in tqdm(datable[enhanced_key_1]):
                    enhanced_sentence_dict = self.enhance_sentence(sentence=sentence,
                                                                   return_entity_desc=return_entity_desc,
                                                                   return_entity_embedding=return_entity_embedding,
                                                                   return_entity_kg=return_entity_kg)
                    enhanced_dict.update(enhanced_sentence_dict)
                save_pickle(enhanced_dict, os.path.join(self.cache_path_file, dict_name))
            else:
                for sentence, sentence_pair in tqdm(zip(datable[enhanced_key_1], datable[enhanced_key_2]),
                                                    total=len(datable[enhanced_key_1])):
                    enhanced_sentence_pair = self.enhance_sentence_pair(sentence=sentence,
                                                                        sentence_pair=sentence_pair,
                                                                        return_entity_desc=return_entity_desc,
                                                                        return_entity_embedding=return_entity_embedding,
                                                                        return_entity_kg=return_entity_kg)
                    enhanced_dict.update(enhanced_sentence_pair)
                save_pickle(enhanced_dict, os.path.join(self.cache_path_file, dict_name))
        return enhanced_dict

    def enhance_train(self,
                      datable,
                      enhanced_key_1="sentence",
                      enhanced_key_2=None,
                      return_entity_desc=False,
                      return_entity_embedding=False,
                      return_entity_kg=False):
        return self._enhance_data(datable=datable,
                                  dict_name="enhanced_train.pkl",
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_entity_desc=return_entity_desc,
                                  return_entity_embedding=return_entity_embedding,
                                  return_entity_kg=return_entity_kg)

    def enhance_dev(self,
                    datable,
                    enhanced_key_1="sentence",
                    enhanced_key_2=None,
                    return_entity_desc=False,
                    return_entity_embedding=False,
                    return_entity_kg=False):
        return self._enhance_data(datable=datable,
                                  dict_name="enhanced_dev.pkl",
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_entity_desc=return_entity_desc,
                                  return_entity_embedding=return_entity_embedding,
                                  return_entity_kg=return_entity_kg)

    def enhance_test(self,
                     datable,
                     enhanced_key_1="sentence",
                     enhanced_key_2=None,
                     return_entity_desc=False,
                     return_entity_embedding=False,
                     return_entity_kg=False):
        return self._enhance_data(datable=datable,
                                  dict_name="enhanced_test.pkl",
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_entity_desc=return_entity_desc,
                                  return_entity_embedding=return_entity_embedding,
                                  return_entity_kg=return_entity_kg)


if __name__ == "__main__":
    from cogktr.data.reader.sst2_reader import Sst2Reader

    enhancer = WorldEnhancer(knowledge_graph_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph",
                             cache_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/enhanced_data",
                             cache_file="world_data",
                             reprocess=False,
                             load_entity_desc=True,
                             load_entity_embedding=False,
                             load_entity_kg=False)

    enhanced_sentence_dict_1 = enhancer.enhance_sentence(sentence="Bert likes reading in the Sesame Street Library.",
                                                         return_entity_desc=True,
                                                         return_entity_embedding=False,
                                                         return_entity_kg=False)
    #
    # enhanced_sentence_dict_2 = enhancer.enhance_sentence(sentence=["Bert", "likes", "reading", "in the", "Street"],
    #                                                      return_entity_desc=True,
    #                                                      return_entity_embedding=True,
    #                                                      return_entity_kg=False)

    # enhanced_sentence_dict_3 = enhancer.enhance_sentence_pair(
    #     sentence="Bert likes reading in the Sesame Street Library.",
    #     sentence_pair="Bert likes reading.",
    #     return_entity_desc=True,
    #     return_entity_embedding=True,
    #     return_entity_kg=False)
    #
    # enhanced_sentence_dict_4 = enhancer.enhance_sentence_pair(
    #     sentence=["Bert", "likes", "reading", "in the", "Street"],
    #     sentence_pair=["Bert", "likes", "reading"],
    #     return_entity_desc=True,
    #     return_entity_embedding=True,
    #     return_entity_kg=False)

    # reader_5 = Sst2Reader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/raw_data")
    # train_data_5, dev_data_5, test_data_5 = reader_5.read_all()
    # enhanced_dev_dict_5 = enhancer.enhance_dev(datable=dev_data_5,
    #                                            enhanced_key_1="sentence",
    #                                            return_entity_desc=True,
    #                                            return_entity_embedding=False,
    #                                            return_entity_kg=False)

    # reader_6 = QnliReader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/sentence_pair/QNLI/raw_data")
    # train_data_6, dev_data_6, test_data_6 = reader_6.read_all()
    # enhanced_dev_dict_6 = enhancer.enhance_dev(datable=dev_data_6,
    #                                            enhanced_key_1="question",
    #                                            enhanced_key_2="sentence",
    #                                            return_ner=True,
    #                                            return_srl=True,
    #                                            return_syntax=True,
    #                                            return_wordnet=True)

    print("end")
