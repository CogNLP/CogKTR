from cogktr.models.base_model import BaseModel
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class SgnetModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.bert = BertModel.from_pretrained(plm)
        self.input_size = 768
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

    def loss(self, batch, loss_function):
        token, label = self.get_batch(batch)
        pred = self.forward(token=token)
        loss = loss_function(pred, label)
        return loss

    def forward(self, token):
        x = self.bert(token).pooler_output
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        token, label = self.get_batch(batch)
        pred = self.predict(token)
        metric_function.evaluate(pred, label)

    def predict(self, token):
        pred = self.forward(token)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

    def get_batch(self, batch):
        token = batch["token"]
        label = batch["label"]
        return token, label