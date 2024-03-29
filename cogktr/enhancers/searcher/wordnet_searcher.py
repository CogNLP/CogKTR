from cogktr.enhancers.searcher import BaseSearcher


class WordnetSearcher(BaseSearcher):
    def __init__(self,
                 tool,
                 return_synset=False,
                 return_synonym=False,
                 return_hypernym=False,
                 return_examples=False,
                 return_definition=False):
        super().__init__()
        if tool not in ["nltk"]:
            raise ValueError("{} in WordnetSearcher is not supported! Please set tool='nltk'".format(tool))
        self.tool = tool
        self.return_synset = return_synset
        self.return_synonym = return_synonym
        self.return_hypernym = return_hypernym
        self.return_examples = return_examples
        self.return_definition = return_definition

    def search(self, lemma_item):
        search_dict = {}
        if self.tool == "nltk":
            search_dict = self._nltk_search(lemma_item)
        return search_dict

    def _nltk_search(self, lemma_item):
        search_dict = {}
        search_dict["lemma_key"] = lemma_item.key()
        if self.return_synonym:
            search_dict["synonym"] = []
            for lemma in lemma_item.synset().lemmas():
                search_dict["synonym"].append(lemma._name)
        if self.return_synset:
            search_dict["synset"] = lemma_item.synset()._name
        if self.return_hypernym:
            # we use the first hypernym synset as the only hypernym
            search_dict["hypernym"] = {}
            if len(lemma_item.synset().hypernyms()) > 0:
                search_dict["hypernym"]["synset"] = lemma_item.synset().hypernyms()[0]._name
                search_dict["hypernym"]["definition"] = lemma_item.synset().hypernyms()[0].definition()
                search_dict["hypernym"]["examples"] = lemma_item.synset().hypernyms()[0].examples()
            else:
                search_dict["hypernym"]["synset"] = None
                search_dict["hypernym"]["definition"] = None
                search_dict["hypernym"]["examples"] = None
        if self.return_examples:
            search_dict["examples"] = lemma_item.synset().examples()
        if self.return_definition:
            search_dict["definition"] = lemma_item.synset().definition()
        return search_dict


if __name__ == "__main__":
    from nltk.corpus import wordnet as wn

    searcher = WordnetSearcher(tool="nltk",
                               return_synset=True,
                               return_synonym=True,
                               return_hypernym=True,
                               return_examples=True,
                               return_definition=True
                               )
    search_dict = searcher.search(wn.lemmas("refer")[0])
    print("end")
