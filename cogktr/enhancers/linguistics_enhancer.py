import os

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
from cogktr.enhancers.base_enhancer import BaseEnhancer
from cogktr.enhancers.tagger.ner_tagger import NerTagger
from cogktr.enhancers.tagger.srl_tagger import SrlTagger
from cogktr.enhancers.tagger.syntax_tagger import SyntaxTagger
from cogktr.enhancers.linker.wordnet_linker import WordnetLinker
from cogktr.enhancers.searcher.wordnet_searcher import WordnetSearcher
from nltk.tokenize import word_tokenize


class LinguisticsEnhancer(BaseEnhancer):
    def __init__(self,
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

    def enhancer_sentence(self, sentence,
                          return_ner=False,
                          return_srl=False,
                          return_syntax=False,
                          return_wordnet=False):
        enhanced_dict = {}

        words = word_tokenize(sentence)
        enhanced_dict["words"] = words

        if return_ner:
            enhanced_dict["ner"] = self.ner_tagger.tag(words)["knowledge"]
        if return_srl:
            enhanced_dict["srl"] = self.srl_tagger.tag(words)["knowledge"]
        if return_syntax:
            enhanced_dict["syntax"] = self.syntax_tagger.tag(words)["knowledge"]
        if return_wordnet:
            wordnet_link_list = self.wordnet_linker.link(words)["knowledge"]
            lemma_dict = {}
            for i, wordnet_link in enumerate(wordnet_link_list):
                lemma_items = wordnet_link["lemma_item"]
                for lemma_item in lemma_items:
                    lemma_dict[lemma_item] = self.wordnet_searcher.search(lemma_item)
                wordnet_link_list[i]["lemma_item_details"] = lemma_dict
            enhanced_dict["wordnet"] = wordnet_link_list

        return enhanced_dict

    def enhancer_sentence_pair(self, sentence, sentence_pair,
                               return_ner=False,
                               return_srl=False,
                               return_syntax=False,
                               return_wordnet=False):
        enhanced_dict = {}
        if return_ner:
            enhanced_dict["ner"] = {}
        if return_srl:
            enhanced_dict["srl"] = {}
        if return_syntax:
            enhanced_dict["syntax"] = {}
        if return_wordnet:
            enhanced_dict["wordnet"] = {}

        return enhanced_dict

    def _enhance_data(self, datable):
        pass

    def enhance_train(self, datable):
        pass

    def enhance_dev(self, datable):
        pass

    def enahcne_test(self, datable):
        pass


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    enhancer = LinguisticsEnhancer(load_ner=True,
                                   load_srl=True,
                                   load_syntax=True,
                                   load_wordnet=True)
    enhanced_sentence_dict_1 = enhancer.enhancer_sentence(sentence="Bert likes reading in the Sesame Street Library.",
                                                          return_ner=True,
                                                          return_srl=True,
                                                          return_syntax=True,
                                                          return_wordnet=True)
    enhanced_sentence_dict_2 = enhancer.enhancer_sentence(sentence=["Bert", "likes", "reading", "in the", "Street"])
    # enhanced_sentence_dict_3 = enhancer.enhancer_sentence_pair(
    #     sentence="Bert likes reading in the Sesame Street Library.",
    #     sentence_pair="Bert likes reading.")
    # enhanced_sentence_dict_4 = enhancer.enhancer_sentence_pair(
    #     sentence=["Bert", "likes", "reading", "in the", "Street"],
    #     sentence_pair=["Bert", "likes", "reading"])
    print("end")
