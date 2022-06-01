class Enhancer:
    def __init__(self,
                 tagger=None,
                 linker=None,
                 searcher=None,
                 embedder=None):

        self.tagger_list=self._to_list(tagger)
        self.linker_list=self._to_list(linker)
        self.searcher_list=self._to_list(searcher)
        self.embedder_list=self._to_list(embedder)

        self.discrete_knowledge={}
        self.discrete_knowledge["ner"]= {}
        self.discrete_knowledge["wikipedia"]= {}
        self.discrete_knowledge["wikipedia"]["entity_title"]= {}
        self.discrete_knowledge["wikipedia"]["neighbor_entity_title"]= {}
        self.continuous_knowledge={}
        self.continuous_knowledge["wikipedia"]= {}
        self.continuous_knowledge["wikipedia"]["entity_embedding"]= {}
        self.continuous_knowledge["wikipedia"]["neighbor_entity_embedding"]= {}


    def _to_list(self,module):
        module_list=[]
        if not isinstance(module,list) and module is not None:
            module_list.append(module)
        if isinstance(module,list):
            module_list=module
        return module_list

    def get_discrete_knowledge(self,sentence):
        for tagger in self.tagger_list:
            self.discrete_knowledge[tagger.knowledge_type]=tagger.tag(sentence)

        for linker in self.linker_list:
            self.discrete_knowledge[linker.knowledge_type]=linker.link(sentence)


        for searcher in self.searcher_list:
            for metion,value_dict in self.discrete_knowledge["wikipedia"].items():
                self.discrete_knowledge["wikipedia"][metion]["desc"]=searcher.search(value_dict["id"])[value_dict["id"]]["desc"]
        return self.discrete_knowledge



    def get_continuous_knowledge(self,sentence):
        for linker in self.linker_list:
            self.continuous_knowledge[linker.knowledge_type]=linker.link(sentence)

        for embedder in self.embedder_list:
            for metion,value_dict in self.continuous_knowledge["wikipedia"].items():
                self.continuous_knowledge["wikipedia"][metion]["entity_embedding"]=embedder.embed(value_dict["title"])[value_dict["title"]]["entity_embedding"]
                self.continuous_knowledge["wikipedia"][metion]["neighbor_entity_embedding"]=embedder.embed(value_dict["title"])[value_dict["title"]]["similar_entities"]
        return self.continuous_knowledge

    def info(self):
        pass

    def help(self):
        pass


if __name__ == "__main__":
    from cogktr import *
    tagger = NerTagger(tool="flair")
    linker = WikipediaLinker(tool="tagme")
    searcher = WikipediaSearcher(tool="blink",
                                 path="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia/raw_data/entity.jsonl")
    embedder = WikipediaEmbedder(tool="wikipedia2vec",
                                 path="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl")

    enhancer=Enhancer(tagger=tagger,
                      linker=linker,
                      searcher=searcher,
                      embedder=embedder)

    sentence="Bert likes reading in the Sesame Street Library."
    discrete_knowledge_dict=enhancer.get_discrete_knowledge(sentence)
    continuous_knowledge_dict=enhancer.get_continuous_knowledge(sentence)
    enhancer.info()
    enhancer.help()
    print("end")
