from cogktr.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch


class BaseSentencePairClassificationModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.input_size = self.plm.hidden_dim
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)
        loss = loss_function(pred, batch["label"])
        return loss

    def forward(self, batch):
        x = self.plm(batch).pooler_output
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


class BaseSentencePairRegressionModel(BaseModel):
    def __init__(self, plm):
        super().__init__()
        self.plm = plm

        self.input_size = self.plm.hidden_dim
        self.linear = nn.Linear(in_features=self.input_size, out_features=1)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)
        pred = pred.squeeze()  # shape:(B,1)->(B)
        loss = loss_function(pred, batch["label"])
        return loss

    def forward(self, batch):
        x = self.plm(batch).pooler_output
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["label"])

    def predict(self, batch):
        pred = self.forward(batch)
        pred = pred.squeeze()  # shape:(B,1)->(B)
        return pred

    def get_batch(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["score"]
        return input_ids, token_type_ids, attention_mask, label
