from cogktr.enhancers.linker import *
import tagme


class WikipediaLinker(BaseLinker):
    def __init__(self, tool, lang="en"):
        super().__init__()
        if tool not in ["tagme"]:
            raise ValueError("{} in WikipediaLinker is not supported!".format(tool))
        self.knowledge_type = "wikipedialinker"
        self.tool = tool
        self.lang = lang

    def link(self, sentence, threshold=0.1):
        link_list = []
        if self.tool == "tagme":
            link_list = self._tagme_link(sentence, threshold)

        return link_list

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
    linker = WikipediaLinker(tool="tagme")
    link_list = linker.link("Bert likes reading in the Sesame Street Library.")
    print("end")
