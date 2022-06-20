# -*- encoding:utf-8 -*-
import torch.nn as nn
from cogktr.modules.layers.layer_norm import LayerNorm
from cogktr.modules.layers.position_ffn import PositionwiseFeedForward
from cogktr.modules.layers.multi_headed_attn import MultiHeadedAttention


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, hidden_size, heads_num, dropout, feedforward_size):
        super(TransformerLayer, self).__init__()

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            hidden_size, heads_num, dropout
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = LayerNorm(hidden_size)
        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            hidden_size, feedforward_size
        )
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_2 = LayerNorm(hidden_size)

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)  
        return output
