from cogktr.enhancers.embedder import BaseEmbedder
from wikipedia2vec import Wikipedia2Vec
import numpy as np

import torch
from cogkge import *
from cogkge.data.lut import LookUpTable
import pickle


class WikipediaEmbedder(BaseEmbedder):
    def __init__(self, tool, path, vocab_path=None, return_entity_embedding=True, return_similar_entities=False,
                 return_similar_entities_num=10):
        super().__init__()
        if tool not in ["wikipedia2vec", "cogkge"]:
            raise ValueError("{} in WikipediaEmbedder is not supported!".format(tool))
        self.tool = tool
        self.path = path
        self.vocab_path = vocab_path
        self.return_entity_embedding = return_entity_embedding
        self.return_similar_entities = return_similar_entities
        self.return_similar_entities_num = return_similar_entities_num

        if self.tool == "wikipedia2vec":
            self.wiki2vec = Wikipedia2Vec.load(path)
        if self.tool == "cogkge":
            with open(self.vocab_path, "rb") as file:
                node_vocab, relation_vocab = pickle.load(file)
            self.node_lut = LookUpTable()
            self.node_lut.add_vocab(node_vocab)
            relation_lut = LookUpTable()
            relation_lut.add_vocab(relation_vocab)
            model = TransE(entity_dict_len=len(self.node_lut),
                           relation_dict_len=len(relation_lut),
                           embedding_dim=100,
                           p_norm=1)
            model.load_state_dict(torch.load(path))
            self.model = model.to("cuda:0")

    def embed(self, title=None, id=None):
        embed_dict = {}
        if self.tool == "wikipedia2vec":
            embed_dict = self._wikipedia2vec_embed(title)
        if self.tool == "cogkge":
            embed_dict = self._cogkge_embed(id)
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

    def _cogkge_embed(self, id):
        embed_dict = {}
        embed_dict["entity_embedding"] = np.zeros(100)
        embed_dict["entity_embedding"] = self.model.e_embedding(
            torch.tensor(self.node_lut.vocab.word2idx[id]).to("cuda:0")).detach().cpu().numpy()
        return embed_dict


if __name__ == "__main__":
    embedder_1 = WikipediaEmbedder(tool="wikipedia2vec",
                                   path="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl")
    embedder_dict_1 = embedder_1.embed(title="Apple")

    embedder_2 = WikipediaEmbedder(tool="cogkge",
                                   path="/data/mentianyi/code/CogKGE/dataset/WIKIPEDIA5M/experimental_output/TransE2022-06-04--19-20-22.14--500epochs/checkpoints/TransE_500epochs/Model.pkl",
                                   vocab_path="/data/mentianyi/code/CogKGE/dataset/WIKIPEDIA5M/processed_data/vocab.pkl")
    embedder_dict_2 = embedder_2.embed(id=18978754)
    print("end")
