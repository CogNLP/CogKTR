from cogktr.models.base_model import BaseModel
import torch.nn as nn
from cogktr.modules.encoder.transformers_syntaxbert import SyntaxBertModel
import torch.nn.functional as F
import torch


class SyntaxAttentionModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.bert = SyntaxBertModel.from_pretrained(plm)
        self.input_size = 768
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

    def loss(self, batch, loss_function):
        input_ids, attention_masks, segment_ids, labels = self.get_batch(batch)
        pred = self.forward(input_ids=input_ids, attention_masks=attention_masks, segment_ids=segment_ids)
        loss = loss_function(pred, labels)
        return loss

    def forward(self, input_ids, attention_masks, segment_ids):
        x = self.bert(input_ids=input_ids, attention_mask=attention_masks,token_type_ids=segment_ids).pooler_output
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        input_ids, attention_masks, segment_ids, labels = self.get_batch(batch)
        pred = self.predict(input_ids=input_ids, attention_masks=attention_masks, segment_ids=segment_ids)
        metric_function.evaluate(pred, labels)

    def predict(self, input_ids, attention_masks, segment_ids):
        pred = self.forward(input_ids=input_ids, attention_masks=attention_masks, segment_ids=segment_ids)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

    def get_batch(self, batch):
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_masks"]
        segment_ids = batch["segment_ids"]
        labels = batch["labels"]
        return input_ids, attention_masks, segment_ids, labels
