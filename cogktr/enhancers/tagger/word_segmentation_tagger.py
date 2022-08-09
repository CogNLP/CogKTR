from cogktr.enhancers.tagger import BaseTagger
import jieba


class WordSegmentationTagger(BaseTagger):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["jieba"]:
            raise ValueError("{} in WordSegmentationTagger is not supported!".format(tool))

        self.tool = tool
        self.knowledge_type = "wordsegmentationtagger"

    def tag(self, sentence):
        tag_dict = {}
        if self.tool == "jieba":
            tag_dict = self._jieba_tag(sentence)

        return tag_dict

    def _jieba_tag(self, sentence):
        tag_dict = {}
        tag_dict["words"] = sentence
        tag_dict["knowledge"] = {}
        tag_dict["knowledge"]["seg"] = list(jieba.cut(sentence, cut_all=False))
        tag_dict["knowledge"]["word_len"] = []
        for seg in tag_dict["knowledge"]["seg"]:
            tag_dict["knowledge"]["word_len"].append(len(seg))

        return tag_dict


if __name__ == "__main__":
    tagger = WordSegmentationTagger(tool="jieba")
    tagger_dict = tagger.tag("伯特喜欢在芝麻街图书馆读书")
    print(tagger_dict)
    print("end")
