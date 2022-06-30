from cogktr.utils.io_utils import load_json
import os
from cogktr.enhancers.tagger import NerTagger, SrlTagger, SyntaxTagger


class NewEnhancer:
    def __init__(self,
                 config_path,
                 load_syntax=False,
                 load_wordnet=False,
                 load_srl=False,
                 load_ner=False,
                 load_conceptnet=False,
                 load_wikipeida=False,
                 load_medicine=False,
                 reprocess=True):
        self.config_path = config_path
        self.load_syntax = load_syntax
        self.load_wordnet = load_wordnet
        self.load_srl = load_srl
        self.load_ner = load_ner
        self.load_conceptnet = load_conceptnet
        self.load_wikipeida = load_wikipeida
        self.load_medicine = load_medicine
        self.reprocess = reprocess

        self.root_path = os.getenv("HOME")
        self.config = load_json(config_path)
        self.syntax_path = os.path.join(self.root_path, self.config["syntax"]["path"])
        self.syntax_tool = self.config["syntax"]["tool"]
        self.ner_path = os.path.join(self.root_path, self.config["ner"]["path"])
        self.ner_tool = self.config["ner"]["tool"]

        if self.load_syntax:
            self.syntax_tagger = SyntaxTagger(tool=self.syntax_tool)
        if self.load_ner:
            self.ner_tagger = NerTagger(tool=self.ner_tool)

        print("end")

    def enhance_sentence(self,
                         sentence,
                         sentence_pair=None,
                         return_syntax=False,
                         return_wordnet=False,
                         return_srl=False,
                         return_ner=False,
                         return_conceptnet=False,
                         return_wikipeida=False,
                         return_medicine=False):
        enhanced_dict = {}
        enhanced_dict[sentence] = {}
        if return_syntax:
            enhanced_dict[sentence]["syntax"] = self.syntax_tagger.tag(sentence)
        if return_ner:
            enhanced_dict[sentence]["ner"] = self.ner_tagger.tag(sentence)

        return enhanced_dict

    def _enhance_data(self,
                      datable,
                      enhanced_key_1,
                      enhanced_key_2,
                      dict_name,
                      return_syntax,
                      return_wordnet,
                      return_srl,
                      return_ner,
                      return_conceptnet,
                      return_wikipeida,
                      return_medicine):
        pass

    def enhance_train(self,
                      datable,
                      enhanced_key_1="sentence",
                      enhanced_key_2=None,
                      return_syntax=False,
                      return_wordnet=False,
                      return_srl=False,
                      return_ner=False,
                      return_conceptnet=False,
                      return_wikipeida=False,
                      return_medicine=False):
        return self._enhance_data(datable=datable,
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  dict_name="enhanced_train_dict.json",
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet,
                                  return_srl=return_srl,
                                  return_ner=return_ner,
                                  return_conceptnet=return_conceptnet,
                                  return_wikipeida=return_wikipeida,
                                  return_medicine=return_medicine)

    def enhance_dev(self,
                    datable,
                    enhanced_key_1="sentence",
                    enhanced_key_2=None,
                    return_syntax=False,
                    return_wordnet=False,
                    return_srl=False,
                    return_ner=False,
                    return_conceptnet=False,
                    return_wikipeida=False,
                    return_medicine=False):
        return self._enhance_data(datable=datable,
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  dict_name="enhanced_dev_dict.json",
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet,
                                  return_srl=return_srl,
                                  return_ner=return_ner,
                                  return_conceptnet=return_conceptnet,
                                  return_wikipeida=return_wikipeida,
                                  return_medicine=return_medicine)

    def enhance_test(self,
                     datable,
                     enhanced_key_1="sentence",
                     enhanced_key_2=None,
                     return_syntax=False,
                     return_wordnet=False,
                     return_srl=False,
                     return_ner=False,
                     return_conceptnet=False,
                     return_wikipeida=False,
                     return_medicine=False):
        return self._enhance_data(datable=datable,
                                  enhanced_key_1=enhanced_key_1,
                                  enhanced_key_2=enhanced_key_2,
                                  dict_name="enhanced_test_dict.json",
                                  return_syntax=return_syntax,
                                  return_wordnet=return_wordnet,
                                  return_srl=return_srl,
                                  return_ner=return_ner,
                                  return_conceptnet=return_conceptnet,
                                  return_wikipeida=return_wikipeida,
                                  return_medicine=return_medicine)


if __name__ == "__main__":
    enhancer = NewEnhancer(config_path="/data/mentianyi/code/CogKTR/cogktr/utils/config/enhancer_config.json",
                           load_syntax=True,
                           load_ner=True)
    enhanced_dict = enhancer.enhance_sentence(sentence="Bert likes reading in the Sesame Street Library.",
                                              return_syntax=True,
                                              return_ner=True)
    print(enhanced_dict)
    print("end")
