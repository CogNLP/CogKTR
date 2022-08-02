import torch.nn as nn
from transformers import AutoModel
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


class KgembKModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self._plm = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = self._plm.embeddings.position_embeddings.embedding_dim
        self.config = self._plm.config_class.from_pretrained(pretrained_model_name)

        self.word_embedding = self._plm.get_input_embeddings().weight
        self.decs_encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.mlp = MLP(input_dim=100, hidden_dim=1000, output_dim=self.hidden_dim)

    def forward(self, batch):
        batch_device = batch["input_ids"].device
        inputs_embeds = self.word_embedding[batch["input_ids"]]
        for i, entity_span in enumerate(batch["entity_span_list"]):
            for entity_dict in entity_span:
                entity_embedding = torch.FloatTensor(entity_dict["entity_embedding"]).to(batch_device)
                entity_embedding = torch.unsqueeze(entity_embedding, dim=0)
                entity_mask = torch.FloatTensor(entity_dict["entity_mask"]).to(batch_device)
                entity_mask = torch.unsqueeze(entity_mask, dim=1)
                entity_embedding = self.mlp(entity_embedding)
                inputs_embeds[i] = entity_mask.mm(entity_embedding) + inputs_embeds[i]

        input_ids = inputs_embeds
        attention_mask = batch["attention_mask"] if "attention_mask" in batch else None
        token_type_ids = batch["token_type_ids"] if "token_type_ids" in batch else None
        return self._plm(inputs_embeds=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
