from cogktr.models.base_model import BaseModel
import torch.nn as nn
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
    def __init__(self, vocab, plm, hidden_dropout_prob):
        super().__init__()
        self.vocab = vocab
        self.plm = plm
        self.hidden_size = self.plm.hidden_dim
        self.hidden_dropout_prob = hidden_dropout_prob

        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.classes_num)

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.c1_to_w1 = GCNLayer(self.hidden_size)
        self.c1_to_c2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w1_to_s1 = GCNLayer(self.hidden_size)
        self.w1_to_w2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.s1_to_w2 = GCNLayer(self.hidden_size)
        self.w2_to_c2 = GCNLayer(self.hidden_size)
        self.relu = nn.ReLU()
        # self.apply(self.init_bert_weights)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)
        loss = loss_function(pred, batch["label"])
        return loss

    def forward(self, batch):
        x = self.plm(batch).last_hidden_state
        char_reps = F.pad(x, [0, 0, 0, batch["c2w"].size(1) - x.size(1)])
        w1 = self.c1_to_w1(torch.transpose(batch["c2w"], -1, -2), char_reps, batch["character_mask"])
        s1 = self.w1_to_s1(torch.transpose(batch["w2s"], -1, -2), w1, batch["word_mask"])
        w2_1 = self.s1_to_w2(batch["w2s"], s1, batch["sentence_mask"])
        w2_2 = self.relu(self.w1_to_w2(w1))
        c2_1 = self.w2_to_c2(batch["c2w"], w2_1 + w2_2, batch["word_mask"])
        c2_2 = self.relu(self.c1_to_c2(char_reps))
        x = torch.sum(c2_1 + c2_2, dim=1)
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["label"])

    def predict(self, batch):
        pred = self.forward(batch)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred
