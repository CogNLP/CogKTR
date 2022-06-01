# pip install allennlp
# pip install allennlp-models

from cogktr.enhancers.tagger import *
from allennlp.predictors.predictor import Predictor

class SrlTagger(BaseTagger):
    def __init__(self,tool):
        super(SrlTagger, self).__init__()
        if tool not in ["allennlp"]:
            raise ValueError("{} in SrlTagger is not supported!".format(tool))
        self.tool = tool
        if self.tool == "allennlp":
            self.srltagger = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    def tag(self,sentence):
        tag_result = self.srltagger.predict(sentence=sentence)
        return tag_result

if __name__ == "__main__":
    tagger = SrlTagger(tool="allennlp")
    tagger_results = tagger.tag("HongbangYuan bought a car from TianyiMen and sold it to ZhuoranJin.")
    print("end")

