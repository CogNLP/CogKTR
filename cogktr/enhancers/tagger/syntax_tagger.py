from cogktr.enhancers.tagger import BaseTagger
import stanza


class SyntaxTagger(BaseTagger):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["stanza"]:
            raise ValueError("{} in PosTagger is not supported!".format(tool))

        self.tool = tool
        self.knowledge_type = "syntaxtagger"
        if self.tool == "stanza":
            # stanza.download(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
            # TODO:Solve download problems and processor parameter problems
            self.syntaxtagger = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

    def tag(self, sentence):
        tag_dict = {}
        if self.tool == "stanza":
            tag_dict = self._stanza_tag(sentence)
        return tag_dict

    def _stanza_tag(self, sentence):
        tag_dict = {}
        word_list = []
        deprel_label_list = []
        head_label_list = []
        indexes_list = []

        tag_result = self.syntaxtagger(sentence)
        for word in tag_result.sentences[0].words:
            word_list.append(word.text)
            deprel_label_list.append(word.deprel)
            head_label_list.append(word.head)
            indexes_list.append(word.id)

        tag_dict["words"] = word_list
        tag_dict["deprel_labels"] = deprel_label_list
        tag_dict["head_labels"] = head_label_list
        tag_dict["indexes"] = indexes_list
        return tag_dict


if __name__ == "__main__":
    tagger = SyntaxTagger(tool="stanza")
    tagger_dict = tagger.tag("Bert likes reading in the Sesame Street Library.")
    print("end")
