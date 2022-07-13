from cogktr.enhancers.linker import BaseLinker
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


class WordnetLinker(BaseLinker):
    def __init__(self, tool):
        super().__init__()
        if tool not in ["nltk"]:
            raise ValueError("{} tool in WordnetLinker is not supported! Please set tool='nltk'".format(tool))
        if tool == "nltk":
            pass

        self.knowledge_type = "wordnetlinker"
        self.tool = tool

    def link(self, sentence, word_index_list=None, word_text_list=None):
        link_dict = {}
        if self.tool == "nltk":
            link_dict = self._nltk_link(sentence, word_index_list, word_text_list)

        return link_dict

    def _nltk_link(self, sentence, word_index_list, word_text_list):
        if isinstance(sentence, str):
            word_index_list = []
            words = word_tokenize(sentence)
            if word_text_list is None:
                word_index_list = list(range(len(words)))
            else:
                for word_text in word_text_list:
                    for index, word in enumerate(words):
                        if word_text == word:
                            word_index_list.append(index)
        elif isinstance(sentence, list):
            words = sentence
            if word_index_list is None:
                word_index_list = list(range(len(words)))
            else:
                word_index_list = word_index_list
        else:
            raise ValueError("Sentence must be str or a list of words!")

        enhancing_word_list = []
        enhancing_word_index_list = []
        for index in word_index_list:
            enhancing_word_list.append(words[index])
            enhancing_word_index_list.append(index)

        link_dict = {}
        link_dict["words"] = words
        link_dict["spans"] = []
        for enhancing_word, enhancing_word_index in zip(enhancing_word_list, enhancing_word_index_list):
            wnl = WordNetLemmatizer()
            word_lemma = wnl.lemmatize(enhancing_word)
            word_lemma_items = wn.lemmas(word_lemma)
            span_dict = {}
            if len(word_lemma_items) > 0:
                span_dict["start_loc"] = enhancing_word_index
                span_dict["end_loc"] = enhancing_word_index + 1
                span_dict["mention"] = enhancing_word
                span_dict["lemma"] = word_lemma
                span_dict["lemma_item"] = []
                for word_lemma_item in word_lemma_items:
                    span_dict["lemma_item"].append(word_lemma_item)
                link_dict["spans"].append(span_dict)
        return link_dict


if __name__ == "__main__":
    linker = WordnetLinker(tool="nltk")
    linker_dict_1 = linker.link(sentence=["Bert", "likes", "reading", "in the Bert Sesame", "Street", "Library."],
                                word_index_list=[0, 1, 5])
    linker_dict_2 = linker.link(sentence="Bert likes reading in the Bert Sesame Street Library.",
                                word_text_list=["Bert", "Street"])
    print(linker_dict_1)
    print(linker_dict_2)
    print("end")
