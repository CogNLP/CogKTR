from cogktr.models.base_model import BaseModel
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_1 = nn.Linear(input_dim, hidden_dim)
        self.act_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.act_1(x)
        x = self.hidden_2(x)
        return x


class KtembModel(BaseModel):
    def __init__(self, vocab, plm,n_epochs):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.bert = BertModel.from_pretrained(plm)
        self.decs_encoder = BertModel.from_pretrained(plm)
        self.word_embedding = self.bert.get_input_embeddings().weight
        self.input_size = 768
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)
        self.mlp = MLP(input_dim=self.input_size, hidden_dim=1000, output_dim=self.input_size)

    def loss(self, batch, loss_function):
        input_ids, attention_masks, segment_ids, valid_masks, label, entity_span_list = self.get_batch(batch)
        pred = self.forward(input_ids=input_ids,
                            attention_masks=attention_masks,
                            segment_ids=segment_ids,
                            valid_masks=valid_masks,
                            entity_span_list=entity_span_list)
        loss = loss_function(pred, label)
        return loss

    def forward(self, input_ids, attention_masks, segment_ids, valid_masks, entity_span_list):
        batch_device = input_ids.device
        inputs_embeds = self.word_embedding[input_ids]
        for i, entity_span in enumerate(entity_span_list):
            for entity_dict in entity_span:
                entity_token = torch.LongTensor(entity_dict["entity_token"]).to(batch_device)
                entity_token = torch.unsqueeze(entity_token, dim=0)
                entity_mask = torch.FloatTensor(entity_dict["entity_mask"]).to(batch_device)
                entity_mask = torch.unsqueeze(entity_mask, dim=1)
                entity_embedding = self.decs_encoder(entity_token).pooler_output
                entity_embedding = self.mlp(entity_embedding)
                inputs_embeds[i] = entity_mask.mm(entity_embedding) + inputs_embeds[i]
        x = self.bert(inputs_embeds=inputs_embeds).pooler_output
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        input_ids, attention_masks, segment_ids, valid_masks, label, entity_span_list = self.get_batch(batch)
        pred = self.predict(input_ids=input_ids,
                            attention_masks=attention_masks,
                            segment_ids=segment_ids,
                            valid_masks=valid_masks,
                            entity_span_list=entity_span_list)
        metric_function.evaluate(pred, label)

    def predict(self, input_ids, attention_masks, segment_ids, valid_masks, entity_span_list):
        pred = self.forward(input_ids=input_ids,
                            attention_masks=attention_masks,
                            segment_ids=segment_ids,
                            valid_masks=valid_masks,
                            entity_span_list=entity_span_list)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

    def get_batch(self, batch):
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_masks"]
        segment_ids = batch["segment_ids"]
        valid_masks = batch["valid_masks"]
        label = batch["label"]
        entity_span_list = batch["entity_span_list"]
        return input_ids, attention_masks, segment_ids, valid_masks, label, entity_span_list
