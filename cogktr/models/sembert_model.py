from cogktr.modules.encoder.sembert import SembertEncoder
from cogktr.models.base_model import BaseModel
import torch.nn as nn
import torch


class SembertForSequenceClassification(BaseModel):
    def __init__(self,vocab,plm,tag_config=None):
        super(SembertForSequenceClassification, self).__init__()
        self.vocab = vocab
        self.num_labels = len(vocab["label_vocab"])
        self.sembert_encoder = SembertEncoder.from_pretrained(
            plm,
            cache_dir="/data/hongbang/.pytorch_pretrained_bert/distributed_-1",
            num_labels=len(vocab["label_vocab"]),
            tag_config=tag_config,
        )
        hidden_size = self.sembert_encoder.hidden_size
        self.pool = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.sembert_encoder.config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_end_idx=None, input_tag_ids=None):
        sequence_output = self.sembert_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_end_idx=start_end_idx,
            input_tag_ids=input_tag_ids,
        )
        first_token_tensor, pool_index = torch.max(sequence_output, dim=1)

        pooled_output = self.pool(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def loss(self,batch,loss_function):
        input_ids, attention_mask, token_type_ids, input_tag_ids, start_end_idx, labels = self.get_batch(batch)
        logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_end_idx=start_end_idx,
            input_tag_ids=input_tag_ids,
        )
        loss = loss_function(logits.view(-1,self.num_labels),labels.view(-1))
        return loss

    def evaluate(self, batch, metric_function):
        input_ids, attention_mask, token_type_ids, input_tag_ids, start_end_idx, labels = self.get_batch(batch)
        logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_end_idx=start_end_idx,
            input_tag_ids=input_tag_ids,
        )
        pred = torch.argmax(logits,dim=-1)
        # print("!!")
        metric_function.evaluate(pred, labels)


    def get_batch(self,batch):
        input_ids,attention_mask,\
        token_type_ids,input_tag_ids,\
        start_end_idx,labels = batch["input_ids"],batch["input_mask"],\
        batch["token_type_ids"],batch["input_tag_ids"],\
        batch["start_end_idx"],batch["label"]

        return input_ids,attention_mask,token_type_ids,input_tag_ids,start_end_idx,labels

if __name__ == '__main__':
    from cogktr import Sst2Reader,Sst2SembertProcessor
    from argparse import Namespace


    reader = Sst2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    processor = Sst2SembertProcessor(plm="bert-base-uncased", max_token_len=128, vocab=vocab, debug=False)
    tag_config = {
        "tag_vocab_size": len(vocab["tag_vocab"]),
        "hidden_size": 10,
        "output_dim": 10,
        "dropout_prob": 0.1,
        "num_aspect": 3
    }
    tag_config = Namespace(**tag_config)
    model = SembertForSequenceClassification(
        vocab=vocab,
        plm="bert-base-uncased",
        tag_config=tag_config
    )


# class BertForSequenceClassificationTag(BaseModel):
#     def __init__(self, vocab, plm, tag_config=None):
#         super().__init__()
#         self.vocab = vocab
#         self.bert = BertModel.from_pretrained(plm)
#         self.num_labels = len(vocab["label_vocab"])
#         self.use_tag = True if tag_config is not None else False
#         config = self.bert.config
#
#         self.filter_size = 3
#         self.cnn = CNN_conv1d(config.hidden_size, config.hidden_dropout_prob, filter_size=self.filter_size)
#
#         self.activation = nn.Tanh()
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         if tag_config is not None:
#             hidden_size = config.hidden_size + tag_config.hidden_size
#             self.tag_model = TagEmbedding(tag_config.tag_vocab_size,tag_config.hidden_size,tag_config.output_dim,tag_config.dropout_prob)
#             self.dense = nn.Linear(tag_config.num_aspect * tag_config.hidden_size, tag_config.hidden_size)
#         else:
#             hidden_size = config.hidden_size
#         if self.use_tag:
#             self.pool = nn.Linear(config.hidden_size + tag_config.hidden_size, config.hidden_size + tag_config.hidden_size)
#             self.classifier = nn.Linear(config.hidden_size + tag_config.hidden_size, self.num_labels)
#         else:
#             self.pool = nn.Linear(config.hidden_size, config.hidden_size)
#             self.classifier = nn.Linear(config.hidden_size, self.num_labels)
#         # self.bert.apply(self.bert.init_bert_weights)
#         self.apply(self.init_bert_weights)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_end_idx=None, input_tag_ids=None):
#         current_device = input_ids.device
#         # sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
#         sequence_output = self.bert(input_ids, token_type_ids, attention_mask)[0]
#         batch_size, sub_seq_len, dim = sequence_output.size()
#         # sequence_output = sequence_output.unsqueeze(1)
#         start_end_idx = start_end_idx  # batch * seq_len * (start, end)
#         max_seq_len = -1
#         max_word_len = self.filter_size
#         for se_idx in start_end_idx:
#             num_words = 0
#             for item in se_idx:
#                 if item[0] != -1 and item[1] != -1:
#                     num_subs = item[1] - item[0] + 1
#                     if num_subs > max_word_len:
#                         max_word_len = num_subs
#                     num_words += 1
#             if num_words > max_seq_len:
#                 max_seq_len = num_words
#         assert max_word_len >= self.filter_size
#         batch_start_end_ids = []
#         batch_id = 0
#         for batch in start_end_idx:
#             word_seqs = []
#             offset = batch_id * sub_seq_len
#             for item in batch:
#                 if item[0] != -1 and item[1] != -1:
#                     subword_ids = list(range(offset + item[0] + 1, offset + item[1] + 2))  # 0用来做padding了
#                     while len(subword_ids) < max_word_len:
#                         subword_ids.append(0)
#                     word_seqs.append(subword_ids)
#             while (len(word_seqs) < max_seq_len):
#                 word_seqs.append([0 for i in range(max_word_len)])
#             batch_start_end_ids.append(word_seqs)
#             batch_id += 1
#
#         batch_start_end_ids = torch.tensor(batch_start_end_ids)
#         batch_start_end_ids = batch_start_end_ids.view(-1)
#         sequence_output = sequence_output.view(-1, dim)
#         sequence_output = torch.cat([sequence_output.new_zeros((1, dim)), sequence_output], dim=0)
#         batch_start_end_ids = batch_start_end_ids.to(current_device)
#         cnn_bert = sequence_output.index_select(0, batch_start_end_ids)
#         cnn_bert = cnn_bert.view(batch_size, max_seq_len, max_word_len, dim)
#         cnn_bert = cnn_bert.to(current_device)
#
#         bert_output = self.cnn(cnn_bert, max_word_len)
#
#         if self.use_tag:
#             num_aspect = input_tag_ids.size(1)
#             input_tag_ids = input_tag_ids[:,:,:max_seq_len]
#             flat_input_tag_ids = input_tag_ids.view(-1, input_tag_ids.size(-1))
#             # print("flat_que_tag", flat_input_que_tag_ids.size())
#             tag_output = self.tag_model(flat_input_tag_ids, num_aspect)
#             # batch_size, que_len, num_aspect*tag_hidden_size
#             tag_output = tag_output.transpose(1, 2).contiguous().view(batch_size,
#                                                                       max_seq_len, -1)
#             tag_output = self.dense(tag_output)
#             sequence_output = torch.cat((bert_output, tag_output), 2)
#         else:
#             sequence_output = bert_output
#
#         #first_token_tensor = sequence_output[:, 0]
#         first_token_tensor, pool_index = torch.max(sequence_output, dim=1)
#
#         pooled_output = self.pool(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         return logits
#
#
#     def loss(self,batch,loss_function):
#         input_ids, attention_mask, token_type_ids, input_tag_ids, start_end_idx, labels = self.get_batch(batch)
#         logits = self.forward(
#             input_ids=input_ids,
#             token_type_ids=token_type_ids,
#             attention_mask=attention_mask,
#             start_end_idx=start_end_idx,
#             input_tag_ids=input_tag_ids,
#         )
#         loss = loss_function(logits.view(-1,self.num_labels),labels.view(-1))
#         return loss
#
#     def evaluate(self, batch, metric_function):
#         input_ids, attention_mask, token_type_ids, input_tag_ids, start_end_idx, labels = self.get_batch(batch)
#         logits = self.forward(
#             input_ids=input_ids,
#             token_type_ids=token_type_ids,
#             attention_mask=attention_mask,
#             start_end_idx=start_end_idx,
#             input_tag_ids=input_tag_ids,
#         )
#         pred = torch.argmax(logits,dim=-1)
#         # print("!!")
#         metric_function.evaluate(pred, labels)
#
#
#     def get_batch(self,batch):
#         input_ids,attention_mask,\
#         token_type_ids,input_tag_ids,\
#         start_end_idx,labels = batch["input_ids"],batch["input_mask"],\
#         batch["token_type_ids"],batch["input_tag_ids"],\
#         batch["start_end_idx"],batch["label"]
#
#         return input_ids,attention_mask,token_type_ids,input_tag_ids,start_end_idx,labels
#
#     def init_bert_weights(self, module):
#         """ Initialize the weights.
#         """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()