from cogktr.modules.encoder.sembert import SembertEncoder
from cogktr.models.base_model import BaseModel
import torch.nn as nn
import torch


class SembertForSequenceClassification(BaseModel):
    def __init__(self, vocab, plm):
        super(SembertForSequenceClassification, self).__init__()
        self.vocab = vocab
        self.num_labels = len(vocab["label_vocab"])
        self.plm = plm
        hidden_size = self.plm.hidden_size
        self.pool = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.plm.config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_end_idx=None, input_tag_ids=None):
        sequence_output = self.plm(
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

    def loss(self, batch, loss_function):
        input_ids, attention_mask, \
        token_type_ids, input_tag_ids, \
        start_end_idx, labels = batch["input_ids"], batch["input_mask"], \
                                batch["token_type_ids"], batch["input_tag_ids"], \
                                batch["start_end_idx"], batch["label"]
        logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_end_idx=start_end_idx,
            input_tag_ids=input_tag_ids,
        )
        loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def evaluate(self, batch, metric_function):
        input_ids, attention_mask, \
        token_type_ids, input_tag_ids, \
        start_end_idx, labels = batch["input_ids"], batch["input_mask"], \
                                batch["token_type_ids"], batch["input_tag_ids"], \
                                batch["start_end_idx"], batch["label"]

        logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_end_idx=start_end_idx,
            input_tag_ids=input_tag_ids,
        )
        pred = torch.argmax(logits, dim=-1)
        metric_function.evaluate(pred, labels)



class SembertForSequenceScore(BaseModel):
    def __init__(self, vocab, plm):
        super(SembertForSequenceScore, self).__init__()
        self.vocab = vocab
        self.plm = plm
        hidden_size = self.plm.hidden_size
        self.pool = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.plm.config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_end_idx=None, input_tag_ids=None):
        sequence_output = self.plm(
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

    def loss(self, batch, loss_function):
        input_ids, attention_mask, \
        token_type_ids, input_tag_ids, \
        start_end_idx, labels = batch["input_ids"], batch["input_mask"], \
                                batch["token_type_ids"], batch["input_tag_ids"], \
                                batch["start_end_idx"], batch["label"]
        logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_end_idx=start_end_idx,
            input_tag_ids=input_tag_ids,
        )
        loss = loss_function(torch.squeeze(logits),labels)
        return loss

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["label"])

    def predict(self,batch):
        input_ids, attention_mask, \
        token_type_ids, input_tag_ids, \
        start_end_idx, labels = batch["input_ids"], batch["input_mask"], \
                                batch["token_type_ids"], batch["input_tag_ids"], \
                                batch["start_end_idx"], batch["label"]
        logits = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_end_idx=start_end_idx,
            input_tag_ids=input_tag_ids,
        )
        return torch.squeeze(logits)

if __name__ == '__main__':
    from cogktr import Sst2Reader, Sst2SembertProcessor
    from argparse import Namespace

    reader = Sst2Reader(raw_data_path="/data/hongbang/CogKTR/datapath/text_classification/SST_2/raw_data")
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
    )
