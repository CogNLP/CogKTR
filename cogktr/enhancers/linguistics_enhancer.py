from .base_enhancer import BaseEnhancer


class LinguisticsEnhancer(BaseEnhancer):
    def __init__(self):
        super().__init__()

    def enhancer_sentence(self, sentence):
        pass

    def enhancer_sentence_pair(self, sentence, sentence_pair):
        pass

    def _enhance_data(self, datable):
        pass

    def enhance_train(self, datable):
        pass

    def enhance_dev(self, datable):
        pass

    def enahcne_test(self, datable):
        pass


if __name__ == "__main__":
    enhancer = LinguisticsEnhancer()
    enhanced_sentence_dict_1 = enhancer.enhancer_sentence(sentence="Bert likes reading in the Sesame Street Library.")
    enhanced_sentence_dict_2 = enhancer.enhancer_sentence(sentence=["Bert", "likes", "reading", "in the", "Street"])
    enhanced_sentence_dict_3 = enhancer.enhancer_sentence_pair(
        sentence="Bert likes reading in the Sesame Street Library.",
        sentence_pair="Bert likes reading in the Sesame Street Library.")
    enhanced_sentence_dict_4 = enhancer.enhancer_sentence_pair(
        sentence=["Bert", "likes", "reading", "in the", "Street"],
        sentence_pair=["Bert", "likes", "reading", "in the", "Street"])
    print("end")
