from wikipedia2vec import Wikipedia2Vec
from transformers import BertModel
import torch.nn as nn
import torch
from transformers import BertTokenizer

class EbertMappingLoadHelper():
    def __init__(self, bert_vocab_path, wikipedia2vec_emb_path):
        self.bert_vocab_path = bert_vocab_path
        self.wikipedia2vec_emb_path = wikipedia2vec_emb_path
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_word_embedding = self.bert.get_input_embeddings().weight

    def load_data(self):
        wiki2vec_emb_list = []
        bert_emb_list = []
        wiki2vec = Wikipedia2Vec.load(self.wikipedia2vec_emb_path)
        with open(self.bert_vocab_path, mode="r") as reader:
            for line in reader:
                word = line.strip()
                try:
                    wiki2vec_word_vector = wiki2vec.get_word_vector(word).tolist()
                    token_id = self.tokenizer.convert_tokens_to_ids(word)
                    bert_word_vector = self.bert_word_embedding[token_id].tolist()
                    wiki2vec_emb_list.append(wiki2vec_word_vector)
                    bert_emb_list.append(bert_word_vector)
                except:
                    pass

        wiki2vec_embs = torch.Tensor(wiki2vec_emb_list)
        bert_embs  = torch.Tensor(bert_emb_list)
        return wiki2vec_embs, bert_embs

class EbertMappingModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, wiki2vec_embs):
        pred_bert_emb = self.linear(wiki2vec_embs)
        return pred_bert_emb

if __name__ == "__main__":
    bert_vocab_path = "/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikipedia_emb_2_bert_emb/vocab.txt"
    wikipedia2vec_emb_path = "/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl"
    matrix_save_path = "/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikipedia_emb_2_bert_emb/mapping.pt"

    print("Load data...")
    loadhelper = EbertMappingLoadHelper(bert_vocab_path, wikipedia2vec_emb_path)
    wiki2vec_embs, bert_embs = loadhelper.load_data()

    model = EbertMappingModel(input_size=100, output_size=768)
    loss_func = nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

    for epoch in range(500):
        pred_bert_emb = model.forward(wiki2vec_embs)
        loss = loss_func(pred_bert_emb, bert_embs)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if epoch % 100 == 0:
            print("epoch: {}, loss function: {}".format(epoch, loss.item()))

    torch.save(model.linear.weight, matrix_save_path)
    print(model.linear.weight.size())
