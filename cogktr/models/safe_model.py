import argparse

from cogktr.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from cogktr.utils.constant.conceptnet_constants.constants import ONE_HOT_FEATURE_LENGTH
from transformers import AutoModel
import math


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
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
                if activation != 'None':
                    self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class GNN_Meta_Path(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size, mp_fea_size, head_count,
                 dropout=0.1):
        super().__init__()
        self.args = args

        self.metapath_fea_encoder = nn.Sequential(
            MLP(mp_fea_size, 100, 1, 1, 0.1, layer_norm=True,
                activation='None'),
        )
        self.nvec2svec = MLP(1, hidden_size, 1, 1, 0, layer_norm=True, activation='None')

    def forward(self, metapath_feature, metapath_feature_count):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        # _batch_size, _n_nodes = node_type_ids.size()
        # n_node_total = _batch_size * _n_nodes
        # edge_index, edge_type = adj  # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph

        # edge_embeddings, edge_index = self.get_graph_edge_embedding(edge_index, edge_type, node_type_ids, n_node_total)
        # x_placeholder = torch.zeros(n_node_total, 1).to(node_type_ids.device)
        # aggr_out = self.mp_helper(x_placeholder, edge_index, edge_embeddings)
        # aggr_out = self.nvec2svec(aggr_out).view(_batch_size, _n_nodes, -1)

        mp_fea_seq, mp_fea_count = metapath_feature, metapath_feature_count
        mp_fea_seq = self.metapath_fea_encoder(mp_fea_seq)  # (bs, 20, 1)
        mp_fea_count = mp_fea_count.unsqueeze(dim=-1)  # (bs,20,1)
        mp_fea_seq = mp_fea_seq * mp_fea_count  # (bs, 20, 1)
        aggr_out = torch.sum(mp_fea_seq, dim=1)  # (bs,1)
        if self.nvec2svec is not None:
            aggr_out = self.nvec2svec(aggr_out)
        return aggr_out


class Meta_Path_MLP(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size, mp_fea_size, head_count,
                 dropout=0.1):
        super().__init__()
        self.args = args

        if args.use_score_sigmoid:
            self.metapath_fea_encoder = nn.Sequential(
                MLP(mp_fea_size, args.metapath_fea_hid, 1, args.mp_fc_layer, args.dropoutmp, layer_norm=True,
                    activation=args.activation),
                nn.Sigmoid()
            )
            self.nvec2svec = None
        elif args.use_score_mlp:
            self.metapath_fea_encoder = nn.Sequential(
                MLP(mp_fea_size, args.metapath_fea_hid, 1, args.mp_fc_layer, args.dropoutmp, layer_norm=True,
                    activation=args.activation),
            )
            self.nvec2svec = MLP(1, hidden_size, 1, 1, 0, layer_norm=True, activation=args.activation)
        elif args.use_score_sigmoid_mlp:
            self.metapath_fea_encoder = nn.Sequential(
                MLP(mp_fea_size, args.metapath_fea_hid, 1, args.mp_fc_layer, args.dropoutmp, layer_norm=True,
                    activation=args.activation),
                nn.Sigmoid()
            )
            self.nvec2svec = MLP(1, hidden_size, 1, 1, 0, layer_norm=True, activation=args.activation)
        else:
            self.metapath_fea_encoder = nn.Sequential(
                MLP(mp_fea_size, args.metapath_fea_hid, 1, args.mp_fc_layer, args.dropoutmp, layer_norm=True,
                    activation=args.activation),
            )
            self.nvec2svec = None
        # self.k = k
        # self.head_count = head_count
        # self.gnn_layers = nn.ModuleList([GSCLayer(1, self.head_count, self.fuse_type) for _ in range(k)])

        # self.activation = GELU()
        # self.dropout = nn.Dropout(dropout)
        # self.dropout_rate = dropout\

    def forward(self, adj, node_type_ids, metapath_feature, metapath_feature_count):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        # _batch_size, _n_nodes = node_type_ids.size()
        # n_node_total = _batch_size * _n_nodes
        # edge_index, edge_type = adj  # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph

        # edge_embeddings, edge_index = self.get_graph_edge_embedding(edge_index, edge_type, node_type_ids, n_node_total)
        # x_placeholder = torch.zeros(n_node_total, 1).to(node_type_ids.device)
        # aggr_out = self.mp_helper(x_placeholder, edge_index, edge_embeddings)
        # aggr_out = self.nvec2svec(aggr_out).view(_batch_size, _n_nodes, -1)

        mp_fea_seq, mp_fea_count = metapath_feature, metapath_feature_count
        # print(mp_fea_count.shape)
        aggr_out = self.metapath_fea_encoder(mp_fea_count)  # (bs, 1)
        # mp_fea_count = mp_fea_count.unsqueeze(dim=-1) # (bs,20,1)
        # mp_fea_seq = mp_fea_seq*mp_fea_count #(bs, 20, 1)
        # aggr_out = torch.sum(mp_fea_seq,dim=1) # (bs,1)
        if self.nvec2svec is not None:
            aggr_out = self.nvec2svec(aggr_out)
        return aggr_out


class Meta_Path_GNN(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, mp_fea_size, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02):
        super().__init__()
        self.args = args
        self.init_range = init_range

        self.weight_gsc = 1
        self.concept_dim = concept_dim
        self.activation = GELU()

        self.gnn = GNN_Meta_Path(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                 input_size=concept_dim, hidden_size=concept_dim,
                                 output_size=fc_dim, mp_fea_size=mp_fea_size[-1], head_count=n_attention_head,
                                 dropout=p_gnn)

        self.fc = MLP(sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sent_vecs, metapath_fea,metapath_fea_count,):
        """
        sent_vecs: (batch_size, dim_sent)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """

        gnn_output = self.gnn(metapath_fea, metapath_fea_count)

        # Z_vecs = gnn_output[:, 0]  # (batch_size, dim_node)
        Z_vecs = gnn_output

        context_score = self.fc(sent_vecs)

        graph_score = Z_vecs * self.weight_gsc
        scores = [context_score, graph_score]
        try:
            logits = context_score + graph_score
        except:
            print(context_score.shape, graph_score.shape)
        return logits, scores


class TextEncoder(nn.Module):
    def __init__(self, model_name, output_token_states=False, from_checkpoint=None, output_attentions=False, **kwargs):
        super().__init__()
        self.model_type = 'roberta'

        self.output_token_states = output_token_states
        self.output_attentions = output_attentions

        self.module = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True,
                                                output_attentions=True)

        self.sent_dim = self.module.config.n_embd if self.model_type in ('gpt',) else self.module.config.hidden_size

    def forward(self, input_ids,attention_mask,token_type_ids):
        '''

        output_token_states: if True, return hidden states of specific layer and attention masks
        '''


        outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        assert type(outputs) is not tuple
        # all_hidden_states = outputs[-2]
        # all_attentions = outputs[-1]
        all_hidden_states = outputs.hidden_states
        all_attentions = outputs.attentions
        hidden_states = outputs.last_hidden_state

        sent_vecs = outputs.pooler_output
        assert sent_vecs is not None

        if self.output_attentions:
            return sent_vecs, all_attentions
        else:
            return sent_vecs, all_hidden_states


class SAFEModel(nn.Module):
    def __init__(self, args=argparse.Namespace(), model_name='roberta-large', k=2, n_ntype=4, n_etype=19,
                 n_concept=799273, concept_dim=32, concept_in_dim=1024, mp_fea_size=torch.Size([500, 4, 528, 42]),
                 n_attention_head=1,
                 fc_dim=512, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.0,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, encoder_config=None):
        super().__init__()
        if encoder_config is None:
            encoder_config = {"output_attentions": False}
        self.args = args

        encoder_config = {"output_attention": False}
        self.encoder = TextEncoder(model_name, **encoder_config)

        self.decoder = Meta_Path_GNN(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                                     n_concept, concept_dim, concept_in_dim, mp_fea_size, n_attention_head,
                                     fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                                     pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                     init_range=init_range)

    def forward(self, batch, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, 1)
        """

        batch_size, num_choices = batch["input_ids"].size(0),batch["input_ids"].size(1)
        batch["input_ids"] = batch["input_ids"].view(batch_size * num_choices,-1)
        batch["attention_mask"] = batch["attention_mask"].view(batch_size * num_choices, -1)
        batch["token_type_ids"] = batch["token_type_ids"].view(batch_size * num_choices, -1)
        metapath_feature = batch["meta_path_feature"].view(batch_size * num_choices, -1, ONE_HOT_FEATURE_LENGTH)
        metapath_feature_count = batch["meta_path_count"].view(batch_size * num_choices, -1)

        lm_inputs = [batch["input_ids"],batch["attention_mask"],batch["token_type_ids"]]

        sent_vecs, all_hidden_states = self.encoder(*lm_inputs)

        logits, _ = self.decoder(sent_vecs,metapath_feature,metapath_feature_count)

        logits = logits.view(batch_size, num_choices)

        return logits

    def loss(self, batch, loss_function):
        logits  = self.forward(batch)
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

if __name__ == '__main__':
    model = SAFEModel()

# class SAFEModel(BaseModel):
#     def __init__(self, vocab, plm):
#         super(SAFEModel, self).__init__()
#         self.plm = AutoModel.from_pretrained(plm,return_dict=True,output_hidden_states=True,output_attentions=True)
#         self.vocab = vocab
#         self.sent_dim = 1024
#         # self.fc = nn.Linear(in_features=self.sent_dim,out_features=1)
#         # self.fc.weight.data.normal_(mean=0.0,std=0.02)
#         # self.fc.bias.data.zero_()
#         self.fc = MLP(
#             input_size=1024,
#             hidden_size=512,
#             output_size=1,
#             num_layers=0,
#             dropout=0.0,
#             layer_norm=True,
#             activation='gelu'
#         )
#
#         self.metapath_encoder = MLP(
#             input_size=42,
#             hidden_size=100,
#             output_size=1,
#             num_layers=1,
#             dropout=0.1,
#             layer_norm=True,
#             activation=None
#         )
#         self.nvec2svec = MLP(
#             input_size=1,
#             hidden_size=32,
#             output_size=1,
#             num_layers=1,
#             dropout=0,
#             layer_norm=True,
#             activation=None
#         )
#
#     def forward(self, batch):
#         # print("Start Debug!")
#
#         batch_size,num_choices = batch["input_ids"].size(0),batch["input_ids"].size(1)
#         batch["input_ids"] = batch["input_ids"].view(batch_size * num_choices,-1)
#         batch["attention_mask"] = batch["attention_mask"].view(batch_size * num_choices, -1)
#         batch["token_type_ids"] = batch["token_type_ids"].view(batch_size * num_choices, -1)
#         metapath_feature = batch["meta_path_feature"].view(batch_size * num_choices, -1, ONE_HOT_FEATURE_LENGTH)
#         metapath_feature_count = batch["meta_path_count"].view(batch_size * num_choices, -1)
#
#         mp_fea_seq,mp_fea_count = metapath_feature,metapath_feature_count
#         mp_fea_seq = self.metapath_encoder(mp_fea_seq) # (batch_size,528,1)
#         mp_fea_count = mp_fea_count.unsqueeze(dim=-1) # (batch_size,528,1)
#         mp_fea_seq = mp_fea_seq * mp_fea_count
#         aggr_out = torch.sum(mp_fea_seq,dim=1) # (batch_size,1)
#         aggr_out = self.nvec2svec(aggr_out)  # (batch_size,1)
#         graph_score = aggr_out
#
#         # sent_vec = self.plm(batch)
#         sent_vec = self.plm(batch["input_ids"],attention_mask=batch["attention_mask"],token_type_ids=batch["token_type_ids"])
#         sent_vec = sent_vec.pooler_output # (batch_size, 1024)
#         context_score = self.fc(sent_vec) # (batch_size,1)
#
#         logits = context_score + graph_score
#         logits = logits.view(batch_size,num_choices)
#         return logits
#
#     def loss(self, batch, loss_function):
#         logits = self.forward(batch)
#         # print(logits.shape," ",batch["label"].shape)
#         loss = loss_function(logits, batch["label"])
#         return loss
#
#     def evaluate(self, batch, metric_function):
#         pred = self.predict(batch)
#         metric_function.evaluate(pred, batch["label"])
#
#     def predict(self, batch):
#         pred = self.forward(batch)
#         pred = F.softmax(pred, dim=1)
#         pred = torch.max(pred, dim=1)[1]
#         return pred
#
# def gelu(x):
#     """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
#         Also see https://arxiv.org/abs/1606.08415
#     """
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#
# class GELU(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return gelu(x)
#
# class MLP(nn.Module):
#     """
#     Multi-layer perceptron
#
#     Parameters
#     ----------
#     num_layers: number of hidden layers
#     """
#     activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
#
#     def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
#                  init_last_layer_bias_to_zero=False, layer_norm=False, activation=None):
#         super().__init__()
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.batch_norm = batch_norm
#         self.layer_norm = layer_norm
#
#         assert not (self.batch_norm and self.layer_norm)
#
#         self.layers = nn.Sequential()
#         for i in range(self.num_layers + 1):
#             n_in = self.input_size if i == 0 else self.hidden_size
#             n_out = self.hidden_size if i < self.num_layers else self.output_size
#             self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
#             if i < self.num_layers:
#                 self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
#                 if self.batch_norm:
#                     self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
#                 if self.layer_norm:
#                     self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
#                 if activation != None:
#                     self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
#         if init_last_layer_bias_to_zero:
#             self.layers[-1].bias.data.fill_(0)
#         self.init_range=0.02
#         self.apply(self._init_weights)
#
#     def forward(self, input):
#         return self.layers(input)
#
#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=self.init_range)
#             if hasattr(module, 'bias') and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#
# if __name__ == '__main__':
#     from transformers import RobertaModel
#     from cogktr import PlmAutoModel
#
#     plm = PlmAutoModel(pretrained_model_name="roberta-large")
#     model = SAFEModel(plm=plm,vocab=None)
#     batch = {
#         "input_ids":torch.randn(2,4,100),
#         "attention_mask":torch.randn(2,4,100),
#         "token_type_ids": torch.randn(2, 4, 100),
#         "meta_path_feature": torch.randn(2, 4, 528,ONE_HOT_FEATURE_LENGTH),
#         "meta_path_count": torch.randn(2, 4, 528,),
#     }
#     logits = model(batch)
