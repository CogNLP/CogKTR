import os
from cogktr.enhancers.base_enhancer import BaseEnhancer
from cogktr.enhancers.tagger.ner_tagger import NerTagger
from cogktr.enhancers.tagger.srl_tagger import SrlTagger
from cogktr.enhancers.tagger.syntax_tagger import SyntaxTagger
from cogktr.enhancers.linker.wordnet_linker import WordnetLinker
from cogktr.enhancers.searcher.wordnet_searcher import WordnetSearcher
from nltk.tokenize import word_tokenize
from cogktr.utils.io_utils import save_json, load_json
from tqdm import tqdm


class LinguisticsEnhancer(BaseEnhancer):
    def __init__(self,
                 cache_path,
                 cache_file,
                 reprocess=False,
                 load_ner=False,
                 load_srl=False,
                 load_syntax=False,
                 load_wordnet=False,
                 ner_tool="cogie",
                 srl_tool="allennlp",
                 syntax_tool="stanza",
                 wordnet_linker_tool="nltk",
                 wordnet_searcher_tool="nltk"):
        super().__init__()
        if not os.path.exists(cache_path):
            raise FileExistsError("{} PATH does not exist!".format(cache_path))
        if not os.path.exists(os.path.join(cache_path, cache_file)):
            os.makedirs(os.path.join(cache_path, cache_file))

        self.cache_path = cache_path
        self.cache_file = cache_file
        self.cache_path_file = os.path.join(cache_path, cache_file)
        self.reprocess = reprocess
        self.load_ner = load_ner
        self.load_srl = load_srl
        self.load_syntax = load_syntax
        self.load_wordnet = load_wordnet

        self.ner_tool = ner_tool
        self.srl_tool = srl_tool
        self.syntax_tool = syntax_tool
        self.wordnet_linker_tool = wordnet_linker_tool
        self.wordnet_searcher_tool = wordnet_searcher_tool

        if self.load_ner:
            self.ner_tagger = NerTagger(tool=self.ner_tool)
        if self.load_srl:
            self.srl_tagger = SrlTagger(tool=self.srl_tool)
        if self.load_syntax:
            self.syntax_tagger = SyntaxTagger(tool=self.syntax_tool)
        if self.load_wordnet:
            self.wordnet_linker = WordnetLinker(tool=self.wordnet_linker_tool)
            self.wordnet_searcher = WordnetSearcher(tool=self.wordnet_searcher_tool,
                                                    return_synset=True,
                                                    return_synonym=True,
                                                    return_hypernym=True,
                                                    return_examples=True,
                                                    return_definition=True)

    def enhance_sentence(self, sentence,
                         return_ner=False,
                         return_srl=False,
                         return_syntax=False,
                         return_wordnet=False):
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

        if return_ner:
            enhanced_dict[sentence]["ner"] = self.ner_tagger.tag(words)["knowledge"]
        if return_srl:
            enhanced_dict[sentence]["srl"] = self.srl_tagger.tag(words)["knowledge"]
        if return_syntax:
            enhanced_dict[sentence]["syntax"] = self.syntax_tagger.tag(words)["knowledge"]
        if return_wordnet:
            wordnet_link_list = self.wordnet_linker.link(words)["knowledge"]
            for i, wordnet_link in enumerate(wordnet_link_list):
                lemma_dict = {}
                lemma_items = wordnet_link["lemma_item"]
                for lemma_item in lemma_items:
                    lemma_dict[str(lemma_item)] = self.wordnet_searcher.search(lemma_item)
                wordnet_link_list[i]["lemma_item_details"] = lemma_dict
                del wordnet_link_list[i]["lemma_item"]
            enhanced_dict[sentence]["wordnet"] = wordnet_link_list

        return enhanced_dict

    def enhance_sentence_pair(self, sentence, sentence_pair,
                              return_ner=False,
                              return_srl=False,
                              return_syntax=False,
                              return_wordnet=False):
        enhanced_dict = {}
        enhanced_sentence_dict = self.enhance_sentence(sentence=sentence,
                                                       return_ner=return_ner,
                                                       return_srl=return_srl,
                                                       return_syntax=return_syntax,
                                                       return_wordnet=return_wordnet)
        enhanced_sentence_pair_dict = self.enhance_sentence(sentence=sentence_pair,
                                                            return_ner=return_ner,
                                                            return_srl=return_srl,
                                                            return_syntax=return_syntax,
                                                            return_wordnet=return_wordnet)
        sentence_key = list(enhanced_sentence_dict.keys())[0]
        sentence_pair_key = list(enhanced_sentence_pair_dict.keys())[0]
        enhanced_dict[(sentence_key, sentence_pair_key)] = {}
        enhanced_dict[(sentence_key, sentence_pair_key)]["sentence"] = enhanced_sentence_dict[sentence_key]
        enhanced_dict[(sentence_key, sentence_pair_key)]["sentence_pair"] = enhanced_sentence_pair_dict[
            sentence_pair_key]
        enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"] = {}

        if return_ner:
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["ner"] = {}
        if return_srl:
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["srl"] = {}
        if return_syntax:
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["syntax"] = {}
        if return_wordnet:
            enhanced_dict[(sentence_key, sentence_pair_key)]["interaction"]["wordnet"] = {}

        return enhanced_dict

    def _enhance_data(self,
                      datable,
                      dict_name=None,
                      enhanced_key_1=None,
                      enhanced_key_2=None,
                      return_ner=False,
                      return_srl=False,
                      return_syntax=False,
                      return_wordnet=False):
        enhanced_dict = {}
        if not self.reprocess and os.path.exists(os.path.join(self.cache_path_file, dict_name)):
            enhanced_dict = load_json(enhanced_dict)
        else:
            print("Enhancing data...")
            if enhanced_key_2 is None:
                for sentence in tqdm(datable[enhanced_key_1]):
                    enhanced_sentence_dict = self.enhance_sentence(sentence=sentence,
                                                                   return_ner=return_ner,
                                                                   return_srl=return_srl,
                                                                   return_syntax=return_syntax,
                                                                   return_wordnet=return_wordnet)
                    enhanced_dict.update(enhanced_sentence_dict)
                save_json(enhanced_dict, os.path.join(self.cache_path_file, dict_name))
            else:
                for sentence, sentence_pair in tqdm(zip(datable[enhanced_key_1], datable[enhanced_key_2]),
                                                    total=len(datable[enhanced_key_1])):
                    enhanced_sentence_pair = self.enhance_sentence_pair(sentence=sentence,
                                                                        sentence_pair=sentence_pair,
                                                                        return_ner=return_ner,
                                                                        return_srl=return_srl,
                                                                        return_syntax=return_syntax,
                                                                        return_wordnet=return_wordnet)
                    enhanced_dict.update(enhanced_sentence_pair)
                save_json(enhanced_dict, os.path.join(self.cache_path_file, dict_name))
        return enhanced_dict

    def enhance_train(self,
                      datable,
                      enhanced_key_1="sentence",
                      enhanced_key_2=None,
                      return_ner=False,
                      return_srl=False,
                      return_syntax=False,
                      return_wordnet=False):
        return self._enhance_data(datable=datable,
                                  dict_name="enhanced_train.json",
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_ner=return_ner,
                                  return_srl=return_srl,
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet)

    def enhance_dev(self,
                    datable,
                    enhanced_key_1="sentence",
                    enhanced_key_2=None,
                    return_ner=False,
                    return_srl=False,
                    return_syntax=False,
                    return_wordnet=False):
        return self._enhance_data(datable=datable,
                                  dict_name="enhanced_dev.json",
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_ner=return_ner,
                                  return_srl=return_srl,
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet)

    def enahcne_test(self,
                     datable,
                     enhanced_key_1="sentence",
                     enhanced_key_2=None,
                     return_ner=False,
                     return_srl=False,
                     return_syntax=False,
                     return_wordnet=False):
        return self._enhance_data(datable=datable,
                                  dict_name="enhanced_test.json",
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  return_ner=return_ner,
                                  return_srl=return_srl,
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet)


if __name__ == "__main__":
    from cogktr.data.reader.sst2_reader import Sst2Reader
    from cogktr.data.reader.qnli_reader import QnliReader

    enhancer = LinguisticsEnhancer(load_ner=True,
                                   load_srl=True,
                                   load_syntax=True,
                                   load_wordnet=True,
                                   cache_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/enhanced_data",
                                   cache_file="linguistics_data",
                                   reprocess=True)

    enhanced_sentence_dict_1 = enhancer.enhance_sentence(sentence="Bert likes reading in the Sesame Street Library.",
                                                         return_ner=True,
                                                         return_srl=True,
                                                         return_syntax=True,
                                                         return_wordnet=True)

    enhanced_sentence_dict_2 = enhancer.enhance_sentence(sentence=["Bert", "likes", "reading", "in the", "Street"],
                                                         return_ner=True,
                                                         return_srl=True,
                                                         return_syntax=True,
                                                         return_wordnet=True)

    enhanced_sentence_dict_3 = enhancer.enhance_sentence_pair(
        sentence="Bert likes reading in the Sesame Street Library.",
        sentence_pair="Bert likes reading.",
        return_ner=True,
        return_srl=True,
        return_syntax=True,
        return_wordnet=True)

    enhanced_sentence_dict_4 = enhancer.enhance_sentence_pair(
        sentence=["Bert", "likes", "reading", "in the", "Street"],
        sentence_pair=["Bert", "likes", "reading"],
        return_ner=True,
        return_srl=True,
        return_syntax=True,
        return_wordnet=True)

    reader_5 = Sst2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data_5, dev_data_5, test_data_5 = reader_5.read_all()
    enhanced_dev_dict_5 = enhancer.enhance_dev(datable=dev_data_5,
                                               enhanced_key_1="sentence",
                                               return_ner=True,
                                               return_srl=True,
                                               return_syntax=True,
                                               return_wordnet=True)

    reader_6 = QnliReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
    train_data_6, dev_data_6, test_data_6 = reader_6.read_all()
    enhanced_dev_dict_6 = enhancer.enhance_dev(datable=dev_data_6,
                                               enhanced_key_1="question",
                                               enhanced_key_2="sentence",
                                               return_ner=True,
                                               return_srl=True,
                                               return_syntax=True,
                                               return_wordnet=True)

    print("end")
