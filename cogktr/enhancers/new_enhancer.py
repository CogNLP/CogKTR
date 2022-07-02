from cogktr.enhancers.searcher.wikidata_searcher import WikidataSearcher
from cogktr.utils.io_utils import load_json, save_json
import os
from tqdm import tqdm
from cogktr.enhancers.tagger import NerTagger, SrlTagger, SyntaxTagger


class NewEnhancer:
    def __init__(self,
                 config_path,
                 knowledge_graph_path,
                 save_file_name,
                 enhanced_data_path,
                 load_syntax=False,
                 load_wordnet=False,
                 load_srl=False,
                 load_ner=False,
                 load_conceptnet=False,
                 load_wikipedia=False,
                 load_medicine=False,
                 reprocess=False):
        self.config_path = config_path
        self.knowledge_graph_path=knowledge_graph_path
        self.save_file_name = save_file_name
        self.enhanced_data_path = enhanced_data_path
        self.load_syntax = load_syntax
        self.load_wordnet = load_wordnet
        self.load_srl = load_srl
        self.load_ner = load_ner
        self.load_conceptnet = load_conceptnet
        self.load_wikipedia = load_wikipedia
        self.load_medicine = load_medicine
        self.reprocess = reprocess

        # self.root_path = os.getenv("HOME")
        self.root_path = knowledge_graph_path
        self.config = load_json(config_path)
        self.syntax_path = os.path.join(self.root_path, self.config["syntax"]["path"])
        self.syntax_tool = self.config["syntax"]["tool"]
        self.ner_path = os.path.join(self.root_path, self.config["ner"]["path"])
        self.ner_tool = self.config["ner"]["tool"]
        self.wikipedia_linker_tool = self.config["wikipedia"]["wikipedia_linker_tool"]
        self.wikipedia_searcher_tool = self.config["wikipedia"]["wikipedia_searcher_tool"]
        self.wikipedia_searcher_path = self.config["wikipedia"]["wikipedia_searcher_path"]
        self.wikipedia_embedder_tool = self.config["wikipedia"]["wikipedia_embedder_tool"]
        self.wikipedia_embedder_path = self.config["wikipedia"]["wikipedia_embedder_path"]

        if self.load_syntax:
            self.syntax_tagger = SyntaxTagger(tool=self.syntax_tool)
        if self.load_ner:
            self.ner_tagger = NerTagger(tool=self.ner_tool)
        if self.load_wikipedia:
            self.wikipedia_linker = WikipediaLinker(tool=self.wikipedia_linker_tool)
            self.wikipedia_searcher = WikipediaSearcher(tool=self.wikipedia_searcher_tool,
                                                        path=self.wikipedia_searcher_path)
            self.wikipedia_embedder = WikipediaEmbedder(tool=self.wikipedia_embedder_tool,
                                                        path=self.wikipedia_embedder_path)
            self.wikidata_searcher = WikidataSearcher()

        print("end")

    def enhance_sentence(self,
                         sentence,
                         return_syntax=False,
                         return_wordnet=False,
                         return_srl=False,
                         return_ner=False,
                         return_conceptnet=False,
                         return_wikipedia=False,
                         return_medicine=False):
        enhanced_dict = {}
        enhanced_dict[sentence] = {}
        if return_syntax:
            enhanced_dict[sentence]["syntax"] = self.syntax_tagger.tag(sentence)
        if return_ner:
            enhanced_dict[sentence]["ner"] = self.ner_tagger.tag(sentence)
        if return_wikipedia:
            entity_list = self.wikipedia_linker.link(sentence)
            for entity in entity_list:
                entity["description"] = self.wikipedia_searcher.search(entity["wikipedia_id"])
                entity["embedding"] = self.wikipedia_embedder.embed(entity["entity_title"])
                entity["kg"] = self.wikidata_searcher.search(wikipedia_id=entity["wikipedia_id"],
                                                             result_num=10)
            enhanced_dict[sentence]["wikipedia"] = entity_list

        return enhanced_dict

    def _enhance_data(self,
                      datable,
                      enhanced_key,
                      enhanced_key_pair,
                      dict_name,
                      return_syntax,
                      return_wordnet,
                      return_srl,
                      return_ner,
                      return_conceptnet,
                      return_wikipedia,
                      return_medicine):
        if not os.path.exists(self.enhanced_data_path):
            raise FileExistsError("{} doesn't exist".format(self.enhanced_data_path))
        enhanced_path = os.path.join(self.enhanced_data_path, self.save_file_name, dict_name)
        if not os.path.exists(enhanced_path):
            os.makedirs(enhanced_path)

        enhanced_dict = {}
        if not self.reprocess:
            if return_syntax:
                syntax_dict = load_json(os.path.join(enhanced_path, "syntax.json"))
            if return_ner:
                ner_dict = load_json(os.path.join(enhanced_path, "ner.json"))
            if return_wikipedia:
                wikipedia_dict = load_json(os.path.join(enhanced_path, "wikipedia.json"))
            for sentence in tqdm(datable[enhanced_key]):
                enhanced_dict[sentence] = {}
                if return_syntax:
                    enhanced_dict[sentence]["syntax"] = syntax_dict[sentence]["syntax"]
                if return_ner:
                    enhanced_dict[sentence]["ner"] = ner_dict[sentence]["ner"]
                if return_wikipedia:
                    enhanced_dict[sentence]["wikipedia"] = wikipedia_dict[sentence]["wikipedia"]
        else:
            print("Enhancing data...")
            if enhanced_key_pair is None:
                for sentence in tqdm(datable[enhanced_key]):
                    enhanced_sentence = self.enhance_sentence(sentence=sentence,
                                                              return_syntax=return_syntax,
                                                              return_wordnet=return_wordnet,
                                                              return_srl=return_srl,
                                                              return_ner=return_ner,
                                                              return_conceptnet=return_conceptnet,
                                                              return_wikipedia=return_wikipedia,
                                                              return_medicine=return_medicine)
                    enhanced_dict.update(enhanced_sentence)
            if enhanced_key_pair is not None:
                for sentence, sentence_pair in tqdm(zip(datable[enhanced_key], datable[enhanced_key_pair]),
                                                    total=len(datable[enhanced_key])):
                    enhanced_sentence = self.enhance_sentence(sentence=sentence,
                                                              return_syntax=return_syntax,
                                                              return_wordnet=return_wordnet,
                                                              return_srl=return_srl,
                                                              return_ner=return_ner,
                                                              return_conceptnet=return_conceptnet,
                                                              return_wikipedia=return_wikipedia,
                                                              return_medicine=return_medicine)
                    enhanced_sentence_pair = self.enhance_sentence(sentence=sentence_pair,
                                                                   return_syntax=return_syntax,
                                                                   return_wordnet=return_wordnet,
                                                                   return_srl=return_srl,
                                                                   return_ner=return_ner,
                                                                   return_conceptnet=return_conceptnet,
                                                                   return_wikipedia=return_wikipedia,
                                                                   return_medicine=return_medicine)
                    enhanced_dict.update(enhanced_sentence)
                    enhanced_dict.update(enhanced_sentence_pair)

            syntax_dict = {}
            ner_dict = {}
            wikipedia_dict = {}
            for sentence, knowledge_type in enhanced_dict.items():
                if return_syntax:
                    syntax_dict[sentence] = {}
                    syntax_dict[sentence]["syntax"] = enhanced_dict[sentence]["syntax"]
                if return_ner:
                    ner_dict[sentence] = {}
                    ner_dict[sentence]["ner"] = enhanced_dict[sentence]["ner"]
                if return_wikipedia:
                    wikipedia_dict[sentence] = {}
                    wikipedia_dict[sentence]["wikipedia"] = enhanced_dict[sentence]["wikipedia"]

            if return_syntax:
                save_json(syntax_dict, os.path.join(enhanced_path, "syntax.json"))
            if return_ner:
                save_json(ner_dict, os.path.join(enhanced_path, "ner.json"))
            if return_wikipedia:
                save_json(wikipedia_dict, os.path.join(enhanced_path, "wikipedia.json"))

        return enhanced_dict

    def enhance_train(self,
                      datable,
                      enhanced_key="sentence",
                      enhanced_key_pair=None,
                      return_syntax=False,
                      return_wordnet=False,
                      return_srl=False,
                      return_ner=False,
                      return_conceptnet=False,
                      return_wikipedia=False,
                      return_medicine=False):
        return self._enhance_data(datable=datable,
                                  enhanced_key=enhanced_key,
                                  enhanced_key_pair=enhanced_key_pair,
                                  dict_name="enhanced_train",
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet,
                                  return_srl=return_srl,
                                  return_ner=return_ner,
                                  return_conceptnet=return_conceptnet,
                                  return_wikipedia=return_wikipedia,
                                  return_medicine=return_medicine)

    def enhance_dev(self,
                    datable,
                    enhanced_key="sentence",
                    enhanced_key_pair=None,
                    return_syntax=False,
                    return_wordnet=False,
                    return_srl=False,
                    return_ner=False,
                    return_conceptnet=False,
                    return_wikipedia=False,
                    return_medicine=False):
        return self._enhance_data(datable=datable,
                                  enhanced_key=enhanced_key,
                                  enhanced_key_pair=enhanced_key_pair,
                                  dict_name="enhanced_dev",
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet,
                                  return_srl=return_srl,
                                  return_ner=return_ner,
                                  return_conceptnet=return_conceptnet,
                                  return_wikipedia=return_wikipedia,
                                  return_medicine=return_medicine)

    def enhance_test(self,
                     datable,
                     enhanced_key="sentence",
                     enhanced_key_pair=None,
                     return_syntax=False,
                     return_wordnet=False,
                     return_srl=False,
                     return_ner=False,
                     return_conceptnet=False,
                     return_wikipedia=False,
                     return_medicine=False):
        return self._enhance_data(datable=datable,
                                  enhanced_key=enhanced_key,
                                  enhanced_key_pair=enhanced_key_pair,
                                  dict_name="enhanced_test",
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet,
                                  return_srl=return_srl,
                                  return_ner=return_ner,
                                  return_conceptnet=return_conceptnet,
                                  return_wikipedia=return_wikipedia,
                                  return_medicine=return_medicine)


if __name__ == "__main__":
    from cogktr import *

    reader = Sst2Reader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    enhancer = NewEnhancer(config_path="/home/chenyuheng/zhouyuyang/CogKTR/cogktr/utils/config/enhancer_config.json",
                           knowledge_graph_path="/home/chenyuheng/zhouyuyang/CogKTR/cogktr/datapath/knowledge_graph",
                           save_file_name="pre_enhanced_data",
                           enhanced_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/enhanced_data",
                           load_syntax=True,
                           load_ner=True,
                           reprocess=True)

    enhanced_sentence_dict = enhancer.enhance_sentence(sentence="Bert likes reading in the Sesame Street Library.",
                                                       return_syntax=True,
                                                       return_ner=True)

    enhanced_dev_dict = enhancer.enhance_dev(dev_data,
                                             return_syntax=True,
                                             return_ner=True)
    print("end")
