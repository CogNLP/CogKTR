from cogktr.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch


class BaseQuestionAnsweringModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.input_size = self.plm.hidden_dim
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=1)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)
        loss = loss_function(pred, batch["label"])
        return loss

    def forward(self, batch):
        batch_size = batch["input_ids"].shape[0]
        batch_answer_num = batch["input_ids"].shape[1]
        batch["input_ids"] = batch["input_ids"].view(batch_size * batch_answer_num, -1)
        batch["attention_mask"] = batch["attention_mask"].view(batch_size * batch_answer_num, -1)
        batch["token_type_ids"] = batch["token_type_ids"].view(batch_size * batch_answer_num, -1)
        batch["special_tokens_mask"] = batch["special_tokens_mask"].view(batch_size * batch_answer_num, -1)
        x = self.plm(batch).pooler_output
        x = self.linear(x)
        x = x.view(batch_size, batch_answer_num)
        return x

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["label"])

    def predict(self, batch):
        pred = self.forward(batch)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

    def analyze(self,batch):
        pred = self.forward(batch)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[0]
        return pred
