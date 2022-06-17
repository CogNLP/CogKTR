from cogktr.models.base_model import BaseModel
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
import torch.nn.functional as F
import torch
import json
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
import numpy as np

######################################Syntax源码的config###############################################
# class SyntaxBertConfig(PretrainedConfig):
#     pretrained_config_archive_map = modeling_bert.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
#
#     def __init__(self,
#                  vocab_size_or_config_json_file=30522,
#                  hidden_size=768,
#                  num_hidden_layers=12,
#                  num_attention_heads=12,
#                  intermediate_size=3072,
#                  hidden_act="gelu",
#                  hidden_dropout_prob=0.1,
#                  attention_probs_dropout_prob=0.1,
#                  max_position_embeddings=512,
#                  type_vocab_size=2,
#                  initializer_range=0.02,
#                  layer_norm_eps=1e-12,
#                  use_syntax=True,
#                  syntax=None,
#                  **kwargs):
#         super(SyntaxBertConfig, self).__init__(**kwargs)
#         if isinstance(vocab_size_or_config_json_file, str):
#             with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
#                 json_config = json.loads(reader.read())
#             for key, value in json_config.items():
#                 self.__dict__[key] = value
#         elif isinstance(vocab_size_or_config_json_file, int):
#             self.vocab_size = vocab_size_or_config_json_file
#             self.hidden_size = hidden_size
#             self.num_hidden_layers = num_hidden_layers
#             self.num_attention_heads = num_attention_heads
#             self.hidden_act = hidden_act
#             self.intermediate_size = intermediate_size
#             self.hidden_dropout_prob = hidden_dropout_prob
#             self.attention_probs_dropout_prob = attention_probs_dropout_prob
#             self.max_position_embeddings = max_position_embeddings
#             self.type_vocab_size = type_vocab_size
#             self.initializer_range = initializer_range
#             self.layer_norm_eps = layer_norm_eps
#
#             # Syntax encoder layer parameter initializations
#             self.use_syntax = use_syntax,
#             self.syntax = syntax
#         else:
#             raise ValueError(
#                 "First argument must be either a vocabulary size (int)"
#                 "or the path to a pretrained model config file (str)")
#################################################################################################################
# class BertConfig(PretrainedConfig):
#     model_type = "bert"
#
#     def __init__(
#             self,
#             vocab_size=30522,
#             hidden_size=768,
#             num_hidden_layers=12,
#             num_attention_heads=12,
#             intermediate_size=3072,
#             hidden_act="gelu",
#             hidden_dropout_prob=0.1,
#             attention_probs_dropout_prob=0.1,
#             max_position_embeddings=512,
#             type_vocab_size=2,
#             initializer_range=0.02,
#             layer_norm_eps=1e-12,
#             pad_token_id=0,
#             position_embedding_type="absolute",
#             use_cache=True,
#             classifier_dropout=None,
#             **kwargs
#     ):
#         super().__init__(pad_token_id=pad_token_id, **kwargs)
#
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.hidden_act = hidden_act
#         self.intermediate_size = intermediate_size
#         self.hidden_dropout_prob = hidden_dropout_prob
#         self.attention_probs_dropout_prob = attention_probs_dropout_prob
#         self.max_position_embeddings = max_position_embeddings
#         self.type_vocab_size = type_vocab_size
#         self.initializer_range = initializer_range
#         self.layer_norm_eps = layer_norm_eps
#         self.position_embedding_type = position_embedding_type
#         self.use_cache = use_cache
#         self.classifier_dropout = classifier_dropout

from cogktr.modules.encoder.transformers_bert import BertModel


class SyntaxJointFusionBertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class SyntaxJointFusionBertModelConfig(PreTrainedModel):
    pass


class SyntaxJointFusionModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.bert = BertModel.from_pretrained(plm)
        self.input_size = 768
        self.classes_num = len(vocab["tag_label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

    def loss(self, batch, loss_function):
        input_ids, attention_masks, segment_ids, valid_masks, label_ids, label_masks, words_len = self.get_batch(batch)
        pred = self.forward(input_ids=input_ids,
                            attention_masks=attention_masks,
                            segment_ids=segment_ids,
                            valid_masks=valid_masks)
        active_loss = label_masks.view(-1) == 1
        active_pred = pred.view(-1, len(self.vocab["ner_label_vocab"]))[active_loss]
        active_labels = label_ids.view(-1)[active_loss]
        loss = loss_function(active_pred, active_labels)
        return loss

    def forward(self, input_ids, attention_masks, segment_ids, valid_masks):
        current_device = input_ids.device
        x = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=segment_ids).last_hidden_state
        batch_size, max_token_len, embedding_dim = x.shape
        valid_x = torch.zeros(batch_size, max_token_len, embedding_dim, dtype=torch.float, device=current_device)
        for i in range(batch_size):
            pos = 0
            for j in range(max_token_len):
                if valid_masks[i][j].item() == 1:
                    valid_x[i][pos] = x[i][j]
                    pos += 1
        valid_x = self.linear(valid_x)
        return valid_x

    def evaluate(self, batch, metric_function):
        input_ids, attention_masks, segment_ids, valid_masks, label_ids, label_masks, words_len = self.get_batch(batch)
        pred_list = self.predict(input_ids=input_ids,
                                 attention_masks=attention_masks,
                                 segment_ids=segment_ids,
                                 valid_masks=valid_masks,
                                 words_len=words_len)
        label_ids_list = label_ids.tolist()
        for i, item in enumerate(label_ids_list):
            label_ids_list[i] = [self.vocab["ner_label_vocab"].id2label(id) for id in item[1:words_len[i] + 1]]
        pred_eval = []
        label_eval = []
        for pred_item, label_ids_item in zip(pred_list, label_ids_list):
            pred_spans = bio_tag_to_spans(tags=pred_item)
            gold_spans = bio_tag_to_spans(tags=label_ids_item)
            for span in pred_spans:
                if span in gold_spans:
                    pred_eval.append(1)
                    label_eval.append(1)
                    gold_spans.remove(span)
                else:
                    pred_eval.append(1)
                    label_eval.append(0)
            for span in gold_spans:
                pred_eval.append(0)
                label_eval.append(1)
        pred_eval = torch.tensor(pred_eval)
        label_eval = torch.tensor(label_eval)
        metric_function.evaluate(pred_eval, label_eval)

    def predict(self, input_ids, attention_masks, segment_ids, valid_masks, words_len):
        pred = self.forward(input_ids=input_ids,
                            attention_masks=attention_masks,
                            segment_ids=segment_ids,
                            valid_masks=valid_masks)
        pred = F.softmax(pred, dim=2)
        pred = torch.max(pred, dim=2)[1]
        pred_list = pred.tolist()
        for i, item in enumerate(pred_list):
            pred_list[i] = [self.vocab["ner_label_vocab"].id2label(id) for id in item[1:words_len[i] + 1]]
        return pred_list

    def get_batch(self, batch):
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_masks"]
        segment_ids = batch["segment_ids"]
        valid_masks = batch["valid_masks"]
        label_ids = batch["label_ids"]
        label_masks = batch["label_masks"]
        words_len = batch["words_len"]
        return input_ids, attention_masks, segment_ids, valid_masks, label_ids, label_masks, words_len
