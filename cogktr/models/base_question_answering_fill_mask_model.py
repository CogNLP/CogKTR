from cogktr.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch


class BaseQuestionAnsweringFillMaskModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm
        # self.input_size = self.plm.hidden_dim
        # self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

    def loss(self, batch, loss_function):
        return self.plm(batch).loss

    def forward(self, batch):
        x = self.plm(batch).logits
        return x

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["masked_token_id"])

    def predict(self, batch, topk=1):
        output = self.forward(batch)
        result_scores = []
        for i, masked_index in enumerate(batch["masked_index"]):
            result_scores.append(output[i, masked_index, :])
        pred = torch.stack(result_scores, dim=0)
        # print(pred.size())
        # pred = F.softmax(result_scores, dim=1)
        # pred = torch.topk(pred, topk, dim=1)[0]
        return pred
