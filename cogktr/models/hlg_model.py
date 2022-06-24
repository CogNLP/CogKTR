from cogktr.models.base_model import BaseModel
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class GCNLayer(nn.Module):
    def __init__(self, hidden_size):
        super(GCNLayer, self).__init__()
        self.gcn = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, adj, nodes_hidden, nodes_mask):
        scale = adj.sum(dim=-1)
        scale[scale == 0] = 1
        adj = adj / scale.unsqueeze(-1).repeat(1, 1, adj.shape[-1])
        nodes_hidden = nodes_hidden * nodes_mask.unsqueeze(-1)
        nodes_hidden = self.gcn(torch.matmul(adj, nodes_hidden))
        return self.relu(nodes_hidden)


class HLGModel(BaseModel):
    def __init__(self, vocab, plm, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.vocab = vocab
        self.plm = plm
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob

        self.bert = BertModel.from_pretrained(plm)
        self.input_size = 768
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.classes_num)
        self.c1_to_w1 = GCNLayer(self.hidden_size)
        self.c1_to_c2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w1_to_s1 = GCNLayer(self.hidden_size)
        self.w1_to_w2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.s1_to_w2 = GCNLayer(self.hidden_size)
        self.w2_to_c2 = GCNLayer(self.hidden_size)
        self.relu = nn.ReLU()
        # self.apply(self.init_bert_weights)

    def loss(self, batch, loss_function):
        input_ids, c2w, w2s, character_mask, word_mask, sentence_mask, input_mask, label = self.get_batch(batch)
        pred = self.forward(input_ids=input_ids, input_mask=input_mask, c2w=c2w, w2s=w2s,
                            character_mask=character_mask, word_mask=word_mask, sentence_mask=sentence_mask)
        loss = loss_function(pred, label)
        return loss

    def forward(self, input_ids, input_mask, c2w, w2s, character_mask, word_mask, sentence_mask):
        x = self.bert(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
        char_reps = F.pad(x, [0, 0, 0, c2w.size(1) - x.size(1)])
        w1 = self.c1_to_w1(torch.transpose(c2w, -1, -2), char_reps, character_mask)
        s1 = self.w1_to_s1(torch.transpose(w2s, -1, -2), w1, word_mask)
        w2_1 = self.s1_to_w2(w2s, s1, sentence_mask)
        w2_2 = self.relu(self.w1_to_w2(w1))
        c2_1 = self.w2_to_c2(c2w, w2_1 + w2_2, word_mask)
        c2_2 = self.relu(self.c1_to_c2(char_reps))
        x = torch.sum(c2_1 + c2_2, dim=1)
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        input_ids, c2w, w2s, character_mask, word_mask, sentence_mask, input_mask, label = self.get_batch(batch)
        pred = self.predict(input_ids=input_ids, input_mask=input_mask, c2w=c2w, w2s=w2s,
                            character_mask=character_mask, word_mask=word_mask, sentence_mask=sentence_mask)
        metric_function.evaluate(pred, label)

    def predict(self, input_ids, input_mask, c2w, w2s, character_mask, word_mask, sentence_mask):
        pred = self.forward(input_ids=input_ids, input_mask=input_mask, c2w=c2w, w2s=w2s,
                            character_mask=character_mask, word_mask=word_mask, sentence_mask=sentence_mask)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

    def get_batch(self, batch):
        input_ids = batch["input_ids"]
        c2w = batch["c2w"]
        w2s = batch["w2s"]
        character_mask = batch["character_mask"]
        word_mask = batch["word_mask"]
        sentence_mask = batch["sentence_mask"]
        input_mask = batch["input_mask"]
        label = batch["label"]
        return input_ids, c2w, w2s, character_mask, word_mask, sentence_mask, input_mask, label
