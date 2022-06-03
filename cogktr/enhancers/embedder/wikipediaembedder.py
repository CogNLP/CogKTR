from cogktr.enhancers.embedder import *
from wikipedia2vec import Wikipedia2Vec
import numpy as np


class WikipediaEmbedder(BaseEmbedder):
    def __init__(self, tool, path, return_entity_embedding=True, return_similar_entities=False,
                 return_similar_entities_num=10):
        super().__init__()
        if tool not in ["wikipedia2vec"]:
            raise ValueError("{} in WikipediaEmbedder is not supported!".format(tool))
        self.tool = tool
        self.path = path
        self.return_entity_embedding = return_entity_embedding
        self.return_similar_entities = return_similar_entities
        self.return_similar_entities_num = return_similar_entities_num

        self.wiki2vec = Wikipedia2Vec.load(path)

    def embed(self, title):
        embed_dict = {}
        if self.tool == "wikipedia2vec":
            embed_dict = self._wikipedia2vec_embed(title)
        return embed_dict

    def _wikipedia2vec_embed(self, title):
        embed_dict = {}
        embed_dict["entity_embedding"] = np.zeros(100)
        embed_dict["similar_entities"] = []
        if self.return_entity_embedding:
            try:
                embed_dict["entity_embedding"] = np.array(self.wiki2vec.get_entity_vector(title))
            except:
                pass
        if self.return_similar_entities:
            embed_dict["similar_entities"] = []
            similar_rank = 1
            similar_entities_list = self.wiki2vec.most_similar(self.wiki2vec.get_entity(title),
                                                               self.return_similar_entities_num)
            for item in similar_entities_list:
                if hasattr(item[0], "title"):
                    item_dict = {}
                    item_dict["similar_entity_title"] = item[0].title
                    item_dict["similar_rank"] = similar_rank
                    item_dict["similarity"] = item[1]
                    item_dict["similar_entity_embedding"] = np.array(self.wiki2vec.get_entity_vector(item[0].title))
                    embed_dict["similar_entities"].append(item_dict)
                    similar_rank += 1
        return embed_dict


if __name__ == "__main__":
    embedder = WikipediaEmbedder(tool="wikipedia2vec",
                                 path="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl")
    embedder_dict = embedder.embed("Apple")
    print("end")
