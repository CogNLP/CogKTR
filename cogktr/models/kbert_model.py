from transformers import BertModel
from cogktr.models.base_model import BaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class KBertForSequenceClassification(BaseModel):
    def __init__(self, plm, vocab):
        super().__init__()
        self.plm = plm
        self.vocab = vocab
        self.bert = BertModel.from_pretrained(plm)
        self.emb_size = 768
        self.word_embedding = self.bert.get_input_embeddings().weight
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.emb_size, out_features=self.classes_num)

    def loss(self, batch, loss_function):
        input_ids, attention_mask, position_ids, labels = self.get_batch(batch)
        inputs_embeds = self.word_embedding[input_ids]
        pred = self.forward(inputs_embeds=inputs_embeds,
                            attention_mask=torch.cuda.FloatTensor(attention_mask),
                            position_ids=position_ids)
        loss = loss_function(pred, torch.cuda.LongTensor(labels))
        return loss

    def forward(self, inputs_embeds, attention_mask, position_ids):
        x = self.bert.forward(inputs_embeds=inputs_embeds,
                              attention_mask=torch.cuda.FloatTensor(attention_mask),
                              position_ids=position_ids).pooler_output
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        input_ids, attention_mask, position_ids, labels = self.get_batch(batch)
        inputs_embeds = self.word_embedding[input_ids]
        pred = self.predict(inputs_embeds=inputs_embeds,
                            attention_mask=torch.cuda.FloatTensor(attention_mask),
                            position_ids=position_ids)
        metric_function.evaluate(pred, torch.cuda.LongTensor(labels))

    def predict(self, inputs_embeds, attention_mask, position_ids):
        pred = self.forward(inputs_embeds=inputs_embeds,
                            attention_mask=torch.cuda.FloatTensor(attention_mask),
                            position_ids=position_ids)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

    def get_batch(self, batch):
        input_ids, attention_mask, position_ids, labels = batch["token_ids"], batch["vm"], batch["pos"], batch["label"]
        return input_ids, attention_mask, position_ids, labels

