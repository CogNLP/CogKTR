from cogktr.enhancers.tagger import BaseTagger
from allennlp.predictors.predictor import Predictor


class SrlTagger(BaseTagger):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["allennlp"]:
            raise ValueError("{} in SrlTagger is not supported!".format(tool))

        self.tool = tool
        self.knowledge_type = "srltagger"
        if self.tool == "allennlp":
            self.srltagger = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    def tag(self, sentence):
        tag_dict = {}
        if self.tool == "allennlp":
            tag_dict = self._allennlp_tag(sentence)
        return tag_dict

    def _allennlp_tag(self, sentence):
        tag_dict = {}
        token_list = []
        label_dict = {}
        tag_result = self.srltagger.predict(sentence=sentence)
        token_list = tag_result["words"]
        for item in tag_result["verbs"]:
            label_dict[item["verb"]] = item["tags"]

        tag_dict["tokens"] = token_list
        tag_dict["labels"] = label_dict

        return tag_dict


if __name__ == "__main__":
    tagger = SrlTagger(tool="allennlp")
    tag_dict = tagger.tag("HongbangYuan bought a car from TianyiMen and sold it to ZhuoranJin.")
    print("end")
