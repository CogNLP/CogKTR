from cogktr.enhancers.linker import *
import tagme


# TODO: add cogie toolkit
class WikipediaLinker(BaseLinker):
    def __init__(self, tool, path, lang="en"):
        super().__init__()
        if tool not in ["tagme"]:
            raise ValueError("Please set a tool!")
        self.tool = tool
        self.path = path
        self.lang = lang

    def link(self, sentence, threshold=0):
        link_dict = {}
        if self.tool == "tagme":
            link_dict = self._tagme_link(sentence, threshold)

        return link_dict

    def _tagme_link(self, sentence, threshold):
        link_dict = {}
        tagme.GCUBE_TOKEN = "5ffc7520-9f20-4858-aabd-80d7bd1bde2f-843339462"
        entities = tagme.annotate(sentence, lang=self.lang)
        for entity in entities.get_annotations(threshold):
            mention = entity.mention
            link_dict[mention] = {}
            link_dict[mention]["begin"] = entity.begin
            link_dict[mention]["end"] = entity.end
            link_dict[mention]["entity_id"] = entity.entity_id
            link_dict[mention]["entity_title"] = entity.entity_title
            link_dict[mention]["score"] = entity.score

        return link_dict


if __name__ == "__main__":
    linker = WikipediaLinker(tool="tagme",
                             path="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia/raw_data/entity.jsonl")
    link_dict = linker.link("Bert likes reading in the library.")
    print("end")
