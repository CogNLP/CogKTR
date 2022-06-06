from cogktr.enhancers.linker import BaseLinker
import tagme
from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit
from cogie.toolkit.el.el_toolkit import ElToolkit


class WikipediaLinker(BaseLinker):
    def __init__(self, tool, lang="en"):
        super().__init__()
        if tool not in ["tagme","cogie"]:
            raise ValueError("{} in WikipediaLinker is not supported!".format(tool))
        if tool == "cogie":
            self.ner_toolkit = NerToolkit(corpus="conll2003")
            self.tokenize_toolkit = TokenizeToolkit()
            self.el_toolkit = ElToolkit(corpus="wiki")

        self.knowledge_type = "wikipedialinker"
        self.tool = tool
        self.lang = lang

    def link(self, sentence, threshold=0.1):
        link_list = []
        if self.tool == "tagme":
            link_list = self._tagme_link(sentence, threshold)
        elif self.tool == "cogie":
            link_list = self._cogie_link(sentence,threshold)

        return link_list

    def _cogie_link(self,sentence,threshold):
        words = self.tokenize_toolkit.run(sentence)
        ner_result = self.ner_toolkit.run(words)
        el_result = self.el_toolkit.run(ner_result)
        return el_result


    def _tagme_link(self, sentence, threshold):
        link_list = []
        tagme.GCUBE_TOKEN = "5ffc7520-9f20-4858-aabd-80d7bd1bde2f-843339462"
        entities = tagme.annotate(sentence, lang=self.lang)
        for entity in entities.get_annotations(threshold):
            link_dict = {}
            mention = entity.mention
            link_dict["begin"] = entity.begin
            link_dict["end"] = entity.end
            link_dict["id"] = entity.entity_id
            link_dict["title"] = entity.entity_title
            link_dict["score"] = entity.score
            link_dict["mention"] = mention
            link_list.append(link_dict)

        return link_list


if __name__ == "__main__":
    linker = WikipediaLinker(tool="blink")
    link_list = linker.link("Bert likes reading in the Sesame Street Library.")
    print("end")
