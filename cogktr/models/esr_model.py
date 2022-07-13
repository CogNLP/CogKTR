import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel
from cogktr.models.base_model import BaseModel


class RobertaForWsd(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    # override forward
    def forward(self, input_ids, attention_mask, token_type_ids, instance_mask, instance_lens, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        instance_mask = instance_mask.unsqueeze(dim=2).repeat(1, 1, self.hidden_size)
        instance_lens = instance_lens.unsqueeze(dim=1).repeat(1, self.hidden_size)
        preds = (outputs[0] * instance_mask).sum(dim=1) / instance_lens
        preds = torch.cat((outputs[1], preds), dim=1)
        preds = self.dropout(preds)
        logits = self.classifier(preds)
        probs = self.softmax(logits)
        if labels is not None:
            return (self.criterion(logits, labels), probs[:, 1].contiguous())
        else:
            return (probs[:, 1].contiguous(),)


def get_wsd_model(config_name):
    model_dict = {
        'RobertaConfig': RobertaForWsd
    }
    return model_dict[config_name]


class EsrModel(BaseModel):
    def __init__(self, vocab, plm, segment_list):
        super().__init__()
        self.vocab = vocab
        self.plm = plm
        self.segment_list = segment_list

        self.input_size = self.plm.hidden_dim
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)
        self.classifier = nn.Linear(self.input_size * 2, 2)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)
        loss = loss_function(pred, batch["label"])
        return loss

    def forward(self, batch):
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
