from cogktr.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from cogktr.utils.constant.conceptnet_constants.constants import ONE_HOT_FEATURE_LENGTH

class SAFEModel(BaseModel):
    def __init__(self, vocab, plm):
        super(SAFEModel, self).__init__()
        self.plm = plm
        self.vocab = vocab
        self.sent_dim = plm.hidden_dim
        self.fc = nn.Linear(in_features=self.sent_dim,out_features=1)
        self.metapath_encoder = MLP(
            input_size=42,
            hidden_size=100,
            output_size=1,
            num_layers=1,
            dropout=0.1,
            layer_norm=True,
            activation=None
        )
        self.nvec2svec = MLP(
            input_size=1,
            hidden_size=32,
            output_size=1,
            num_layers=1,
            dropout=0,
            layer_norm=True,
            activation=None
        )

    def forward(self, batch):
        # print("Start Debug!")

        batch_size,num_choices = batch["input_ids"].size(0),batch["input_ids"].size(1)
        batch["input_ids"] = batch["input_ids"].view(batch_size * num_choices,-1)
        batch["attention_mask"] = batch["attention_mask"].view(batch_size * num_choices, -1)
        batch["token_type_ids"] = batch["token_type_ids"].view(batch_size * num_choices, -1)
        metapath_feature = batch["meta_path_feature"].view(batch_size * num_choices, -1, ONE_HOT_FEATURE_LENGTH)
        metapath_feature_count = batch["meta_path_count"].view(batch_size * num_choices, -1)

        mp_fea_seq,mp_fea_count = metapath_feature,metapath_feature_count
        mp_fea_seq = self.metapath_encoder(mp_fea_seq) # (batch_size,528,1)
        mp_fea_count = mp_fea_count.unsqueeze(dim=-1) # (batch_size,528,1)
        mp_fea_seq = mp_fea_seq * mp_fea_count
        aggr_out = torch.sum(mp_fea_seq,dim=1) # (batch_size,1)
        aggr_out = self.nvec2svec(aggr_out)  # (batch_size,1)
        graph_score = aggr_out

        sent_vec = self.plm(batch)
        sent_vec = sent_vec.pooler_output # (batch_size, 1024)
        context_score = self.fc(sent_vec) # (batch_size,1)

        logits = context_score + graph_score
        logits = logits.view(batch_size,num_choices)
        return logits

    def loss(self, batch, loss_function):
        logits = self.forward(batch)
        # print(logits.shape," ",batch["label"].shape)
        loss = loss_function(logits, batch["label"])
        return loss

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["label"])

    def predict(self, batch):
        pred = self.forward(batch)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred



class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                if activation != None:
                    self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)



if __name__ == '__main__':
    from transformers import RobertaModel
    from cogktr import PlmAutoModel

    plm = PlmAutoModel(pretrained_model_name="roberta-large")
    model = SAFEModel(plm=plm,vocab=None)
    batch = {
        "input_ids":torch.randn(2,4,100),
        "attention_mask":torch.randn(2,4,100),
        "token_type_ids": torch.randn(2, 4, 100),
        "meta_path_feature": torch.randn(2, 4, 528,ONE_HOT_FEATURE_LENGTH),
        "meta_path_count": torch.randn(2, 4, 528,),
    }
    logits = model(batch)
