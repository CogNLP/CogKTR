from cogktr.enhancers.tagger import *
from flair.models import SequenceTagger
from flair.data import Sentence


class PosTagger(BaseTagger):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["flair"]:
            raise ValueError("{} in PosTagger is not supported!".format(tool))

        self.tool = tool
        self.knowledge_type = "postagger"
        if self.tool == "flair":
            self.nertagger = SequenceTagger.load('pos')

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
        for entity in flair_sentence.get_labels('pos'):
            label_list.append(entity.value)

        tag_dict["tokens"] = token_list
        tag_dict["labels"] = label_list

        return tag_dict


if __name__ == "__main__":
    tagger = PosTagger(tool="flair")
    tagger_dict = tagger.tag("Bert likes reading in the Sesame Street Library.")
    print("end")