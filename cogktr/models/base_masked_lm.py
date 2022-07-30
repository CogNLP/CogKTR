from cogktr.models.base_model import BaseModel
# import torch.nn as nn
# import torch.nn.functional as F
import torch
# from transformers.activations import ACT2FN
from transformers import AutoModelForMaskedLM
# from transformers import AutoModel


# class BertPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         if isinstance(config.hidden_act, str):
#             self.transform_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.transform_act_fn = config.hidden_act
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states
#
#
# class BertLMPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.transform = BertPredictionHeadTransform(config)
#
#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#
#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))
#
#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias
#
#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states
#
#
# class BertOnlyMLMHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.predictions = BertLMPredictionHead(config)
#
#     def forward(self, sequence_output):
#         prediction_scores = self.predictions(sequence_output)
#         return prediction_scores


class BaseMaskedLM(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm
        self.pretrained_model_name = plm.pretrained_model_name
        self.whole_mlm_bert = AutoModelForMaskedLM.from_pretrained(self.pretrained_model_name)

    def forward(self, batch):
        x = self.plm(batch).last_hidden_state
        x = self.whole_mlm_bert.cls(x)
        return x

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["masked_token_id"])

    def predict(self, batch):
        pred = self.forward(batch)
        result_scores = []
        for i, masked_index in enumerate(batch["masked_index"]):
            result_scores.append(pred[i, masked_index, :])
        pred = torch.stack(result_scores, dim=0)
        return pred
