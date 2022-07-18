import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel
from cogktr.models.base_model import BaseModel


class TEsrModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        # self.vocab = vocab
        # self.plm = plm
        #
        # self.input_size = self.plm.hidden_dim
        # self.classes_num = len(vocab["label_vocab"])
        # self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)
        # self.classifier = nn.Linear(self.input_size, 2)

        self.vocab = vocab
        self.plm = plm

        self.input_size = self.plm.hidden_dim
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)
        self.classifier = nn.Linear(self.input_size * 2, 2)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)
        loss = loss_function(pred, batch["label"])
        return loss

    def forward(self, batch):
        # plm_output = self.plm(batch)
        # pooler_output = plm_output.pooler_output
        # logits = self.classifier(pooler_output)
        # return logits

        plm_output = self.plm(batch)
        pooler_output = plm_output.pooler_output
        last_hidden_state = plm_output.last_hidden_state
        instance_mask = batch["instance_mask"].unsqueeze(dim=2).repeat(1, 1, self.input_size)
        instance_lens = batch["instance_lens"].unsqueeze(dim=1).repeat(1, self.input_size)
        preds = (last_hidden_state * instance_mask).sum(dim=1) / instance_lens
        preds = torch.cat((pooler_output, preds), dim=1)
        logits = self.classifier(preds)
        return logits

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["label"])

    def predict(self, batch):
        pred = self.forward(batch)
        pred = pred[:, 1].contiguous()
        return pred
