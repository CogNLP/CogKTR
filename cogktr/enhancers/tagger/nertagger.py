from cogktr.enhancers.tagger import *
from flair.models import SequenceTagger
from flair.data import Sentence


class NerTagger(BaseTagger):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["flair"]:
            raise ValueError("{} in SrlTagger is not supported!".format(tool))

        self.tool = tool
        self.knowledge_type = "nertagger"
        if self.tool == "flair":
            self.nertagger = SequenceTagger.load('ner')

    def tag(self, sentence):
        tag_dict = {}
        if self.tool == "flair":
            tag_dict = self._flair_tag(sentence)
        return tag_dict

    def _flair_tag(self, sentence):
        tag_dict = {}
        token_list = []
        label_list = []
        flair_sentence = Sentence(sentence)
        self.nertagger.predict(flair_sentence)

        for item in flair_sentence.tokens:
            token_list.append(item.form)
            label_list.append("O")
        for entity in flair_sentence.get_spans('ner'):
            for word in entity:
                label_list[word.idx - 1] = entity.tag

        tag_dict["tokens"] = token_list
        tag_dict["labels"] = label_list

        return tag_dict


if __name__ == "__main__":
    tagger = NerTagger(tool="flair")
    tagger_dict = tagger.tag("Bert likes reading in the Sesame Street Library.")
    print("end")
