import torch.cuda
import torch.nn as nn
from transformers import AutoModel


class KbertKModel(nn.Module):
    def __init__(self, pretrained_model_name, device):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self._plm = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = self._plm.embeddings.position_embeddings.embedding_dim
        self.config=self._plm.config_class.from_pretrained(pretrained_model_name)
        self.device = device

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = torch.FloatTensor(batch["attention_mask"]).to(self.device) if "attention_mask" in batch else None
        token_type_ids = batch["token_type_ids"] if "token_type_ids" in batch else None
        position_ids = batch["position_ids"] if "position_ids" in batch else None
        return self._plm(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         position_ids=position_ids)