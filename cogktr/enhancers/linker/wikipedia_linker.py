from cogktr.enhancers.linker.base_linker import BaseLinker
import tagme
from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit
from cogie.toolkit.el.el_toolkit import ElToolkit


class WikipediaLinker(BaseLinker):
    def __init__(self, tool, lang="en"):
        super().__init__()
        if tool not in ["tagme", "cogie"]:
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
        if self.tool == "cogie":
            link_list = self._cogie_link(sentence)

        return link_list

    def _cogie_link(self, sentence):
        if isinstance(sentence, str):
            words = self.tokenize_toolkit.run(sentence)
        elif isinstance(sentence, list):
            words = sentence
        else:
            raise ValueError("Sentence must be str or a list of words!")

        ner_result = self.ner_toolkit.run(words)
        el_result = self.el_toolkit.run(ner_result)
        link_dict = {}
        link_dict["words"] = words
        span_list = []
        for item in el_result:
            span_dict = {}
            span_dict["start"] = item["start"]
            span_dict["end"] = item["end"]
            span_dict["entity_title"] = item["title"]
            span_dict["wikipedia_id"] = int(item["url"].split("=")[1])
            span_list.append(span_dict)
        link_dict["spans"] = span_list
        return link_dict

    def _tagme_link(self, sentence, threshold):
        link_list = []
        tagme.GCUBE_TOKEN = "5ffc7520-9f20-4858-aabd-80d7bd1bde2f-843339462"
        entities = tagme.annotate(sentence, lang=self.lang)
        for entity in entities.get_annotations(threshold):
            link_dict = {}
            link_dict["start"] = entity.begin
            link_dict["end"] = entity.end
            link_dict["wikipedia_id"] = entity.entity_id
            link_dict["entity_title"] = entity.entity_title
            link_list.append(link_dict)

        return link_list


if __name__ == "__main__":
    linker_1 = WikipediaLinker(tool="tagme")
    link_list_1 = linker_1.link("Bert likes reading in the Sesame Street Library.")

    linker_2 = WikipediaLinker(tool="cogie")
    link_list_2 = linker_2.link("Bert likes reading in the Sesame Street Library.")
    link_list_3 = linker_2.link(["Bert", "likes", "reading", "in the Sesame", "Street", "Library."])
    print("end")
