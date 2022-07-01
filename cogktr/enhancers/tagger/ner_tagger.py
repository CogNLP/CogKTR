from cogktr.enhancers.tagger import BaseTagger
from flair.data import Sentence
from flair.models import SequenceTagger
from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit


class NerTagger(BaseTagger):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["flair","cogie"]:
            raise ValueError("{} in NerTagger is not supported!".format(tool))

        self.tool = tool
        self.knowledge_type = "nertagger"
        if self.tool == "flair":
            self.nertagger = SequenceTagger.load('ner')
        elif self.tool == "cogie":
            self.tokenizer_toolkit = TokenizeToolkit()
            self.ner_toolkit = NerToolkit(corpus="conll2003")

    def tag(self, sentence):
        tag_dict = {}
        if self.tool == "flair":
            tag_dict = self._flair_tag(sentence)
        elif self.tool == "cogie":
            tag_dict = self._cogie_tag(sentence)
        return tag_dict

    def _flair_tag(self, sentence):
        flair_sentence=None
        if isinstance(sentence, str):
            flair_sentence = Sentence(sentence, use_tokenizer=True)
        if isinstance(sentence, list):
            flair_sentence = Sentence(sentence, use_tokenizer=False)

        tag_dict = {}
        token_list = []
        label_list = []

        self.nertagger.predict(flair_sentence)
        for item in flair_sentence.tokens:
            token_list.append(item.form)
            label_list.append("O")
        for entity in flair_sentence.get_spans('ner'):
            for word in entity:
                label_list[word.idx - 1] = entity.tag

        tag_dict["words"] = token_list
        tag_dict["ner_labels"] = label_list
        return tag_dict


    def _cogie_tag(self,sentence):
        if isinstance(sentence,str):
            words = self.tokenizer_toolkit.run(sentence)
        elif isinstance(sentence,list):
            words = sentence
        else:
            raise ValueError("Sentence must be str or a list of words!")

        ner_result = self.ner_toolkit.run(words)
        return ner_result



if __name__ == "__main__":
    tagger = NerTagger(tool="cogie")
    tagger_dict_1 = tagger.tag(["Bert", "likes", "reading", "in the Sesame", "Street", "Library."])
    tagger_dict_2 = tagger.tag("Bert likes reading in the Sesame Street Library.")
    print(tagger_dict_1)
    print(tagger_dict_2)
    print("end")
