from cogktr.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch


def bio_tag_to_spans(tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]


class BaseSequenceLabelingModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.input_size = self.plm.hidden_dim
        self.classes_num = len(vocab["ner_label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)
        label_masks = batch["label_masks"]
        label_ids = batch["label_ids"]
        active_loss = label_masks.view(-1) == 1
        active_pred = pred.view(-1, len(self.vocab["ner_label_vocab"]))[active_loss]
        active_labels = label_ids.view(-1)[active_loss]
        loss = loss_function(active_pred, active_labels)
        return loss

    def forward(self, batch):
        input_ids = batch["input_ids"]
        valid_masks = batch["valid_masks"]
        current_device = input_ids.device
        x = self.plm(batch).last_hidden_state
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
        label_ids=batch["label_ids"]
        words_len=batch["words_len"]
        pred_list = self.predict(batch)
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

    def predict(self, batch):
        words_len=batch["words_len"]
        pred = self.forward(batch)
        pred = F.softmax(pred, dim=2)
        pred = torch.max(pred, dim=2)[1]
        pred_list = pred.tolist()
        for i, item in enumerate(pred_list):
            pred_list[i] = [self.vocab["ner_label_vocab"].id2label(id) for id in item[1:words_len[i] + 1]]
        return pred_list
