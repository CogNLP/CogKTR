import torch
import torch.nn as nn
from cogktr.modules.layers.layer_norm import LayerNorm
from cogktr.modules.encoder.bert import BertEncoder
from cogktr.modules.layers.bert_target import BertTarget


class Model(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """

    def __init__(self, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

    def forward(self, src, tgt, seg, pos=None, vm=None):
        # [batch_size, seq_length, emb_size]

        emb = self.embedding(src, seg, pos)

        output = self.encoder(emb, seg, vm)

        loss_info = self.target(output, tgt)

        return loss_info

class KBertClassifier(nn.Module):
    def __init__(self, labels_num, pooling, hidden_size, no_vm, model):
        super(KBertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = labels_num
        self.pooling = pooling
        self.output_layer_1 = nn.Linear(hidden_size, hidden_size)
        self.output_layer_2 = nn.Linear(hidden_size, labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits

class KBertTagger(nn.Module):
    def __init__(self, args, model):
        super(KBertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        output = self.encoder(emb, mask, vm)
        # Target.
        output = self.output_layer(output)

        output = output.contiguous().view(-1, self.labels_num)
        output = self.softmax(output)

        label = label.contiguous().view(-1, 1)
        label_mask = (label > 0).float().to(torch.device(label.device))
        one_hot = torch.zeros(label_mask.size(0), self.labels_num). \
            to(torch.device(label.device)). \
            scatter_(1, label, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        predict = output.argmax(dim=-1)
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )

        return loss, correct, predict, label

class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, dropout, emb_size, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(self.max_length, emb_size)
        self.segment_embedding = nn.Embedding(3, emb_size)
        self.layer_norm = LayerNorm(emb_size)

    def forward(self, src, seg, pos=None):
        word_emb = self.word_embedding(src)
        if pos is None:
            pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                            dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        else:
            pos_emb = self.position_embedding(pos)
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb

class KBertModelBuilder():
    def __init__(self, emb_size, vocab, layers_num, hidden_size, heads_num, feedforward_size, labels_num = 2, pooling="first", no_vm=False, dropout=0.5, ):
        self.model = self.build_bert_model(dropout, emb_size, vocab, layers_num, hidden_size, heads_num, feedforward_size)
        self.labels_num = labels_num
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.no_vm = no_vm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def build_bert_model(self, dropout, emb_size, vocab, layers_num, hidden_size, heads_num, feedforward_size):
        embedding = BertEmbedding(dropout, emb_size, len(vocab))
        encoder = BertEncoder(layers_num, hidden_size, heads_num, dropout, feedforward_size)
        target = BertTarget(hidden_size, len(vocab))
        model = Model(embedding, encoder, target)

        return model

    def load_params(self, pretrained_model_path):
        self.model.load_state_dict(torch.load(pretrained_model_path), strict=False)

    def build_classification_model(self):
        self.model = KBertClassifier(self.labels_num, self.pooling, self.hidden_size, self.no_vm, self.model)

    def batch_loader(self, batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vms_batch = vms[i*batch_size: (i+1)*batch_size]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vms_batch = vms[instances_num//batch_size*batch_size:]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    def set_device(self):
        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    # Evaluation function.
    def evaluate(self, dataset, is_test, metrics='Acc'):

        input_ids = torch.LongTensor(dataset.datable.datas["token_ids"])
        label_ids = torch.LongTensor(dataset.datable.datas["label"])
        mask_ids = torch.LongTensor(dataset.datable.datas["mask"])
        pos_ids = torch.LongTensor(dataset.datable.datas["pos"])
        vms = dataset.datable.datas["vm"]

        batch_size = self.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(self.labels_num, self.labels_num, dtype=torch.long)

        self.model.eval()


        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(
                self.batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(self.device)
            label_ids_batch = label_ids_batch.to(self.device)
            mask_ids_batch = mask_ids_batch.to(self.device)
            pos_ids_batch = pos_ids_batch.to(self.device)
            vms_batch = vms_batch.to(self.device)

            with torch.no_grad():
                try:
                    loss, logits = self.model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch,
                                         vms_batch)
                except:
                    print(input_ids_batch)
                    print(input_ids_batch.size())
                    print(vms_batch)
                    print(vms_batch.size())

            logits = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(logits, dim=1)
            gold = label_ids_batch
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()

        if is_test:
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")

        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / confusion[i, :].sum().item()
            r = confusion[i, i].item() / confusion[:, i].sum().item()
            f1 = 2 * p * r / (p + r)
            if i == 1:
                label_1_f1 = f1
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
        if metrics == 'Acc':
            return correct / len(dataset)
        elif metrics == 'f1':
            return label_1_f1
        else:
            return correct / len(dataset)