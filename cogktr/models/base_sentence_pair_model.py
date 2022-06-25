from cogktr.models.base_model import BaseModel
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class BaseSentencePairClassificationModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.bert = BertModel.from_pretrained(plm)
        self.input_size = 768
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

    def loss(self, batch, loss_function):
        input_ids, token_type_ids, attention_mask, label = self.get_batch(batch)
        pred = self.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = loss_function(pred, label)
        return loss

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        input_ids, token_type_ids, attention_mask, label = self.get_batch(batch)
        pred = self.predict(input_ids, token_type_ids, attention_mask)
        metric_function.evaluate(pred, label)

    def predict(self, input_ids, token_type_ids, attention_mask):
        pred = self.forward(input_ids, token_type_ids, attention_mask)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

    def get_batch(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        return input_ids, token_type_ids, attention_mask, label


class BaseSentencePairRegressionModel(BaseModel):
    def __init__(self, plm):
        super().__init__()
        self.plm = plm

        self.bert = BertModel.from_pretrained(plm)
        self.input_size = 768
        self.linear = nn.Linear(in_features=self.input_size, out_features=1)

    def loss(self, batch, loss_function):
        input_ids, token_type_ids, attention_mask, label = self.get_batch(batch)
        pred = self.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pred = pred.squeeze()  # shape:(B,1)->(B)
        loss = loss_function(pred, label)
        return loss

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        input_ids, token_type_ids, attention_mask, label = self.get_batch(batch)
        pred = self.predict(input_ids, token_type_ids, attention_mask)
        metric_function.evaluate(pred, label)

    def predict(self, input_ids, token_type_ids, attention_mask):
        pred = self.forward(input_ids, token_type_ids, attention_mask)
        pred = pred.squeeze()  # shape:(B,1)->(B)
        return pred

    def get_batch(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["score"]
        return input_ids, token_type_ids, attention_mask, label
