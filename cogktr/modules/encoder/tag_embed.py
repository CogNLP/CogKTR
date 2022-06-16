import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class TagEmbeddings(nn.Module):
    """Simple tag embeddings, randomly initialized."""
    def __init__(self, tag_vocab_size,hidden_size,dropout_prob,padding_idx=0):
        super(TagEmbeddings, self).__init__()
        self.tag_embeddings = nn.Embedding(tag_vocab_size, hidden_size, padding_idx=padding_idx)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tag_ids):
        tags_embeddings = self.tag_embeddings(input_tag_ids)
        embeddings = tags_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TagEmbedding(nn.Module):
    def __init__(self, tag_vocab_size,hidden_size,output_dim,dropout_prob):
        super(TagEmbedding, self).__init__()
        # Embedding
        self.hidden_size = hidden_size
        self.embed = TagEmbeddings(tag_vocab_size,hidden_size,dropout_prob)
        # Linear
        #self.fc = nn.Linear(config.hidden_size * 2, config.output_dim)
        self.fc = nn.Linear(hidden_size, output_dim)
        #  dropout
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, flat_input_ids, num_aspect):  # flat_input_ids.size() = (batch_size*num_aspect, seq_len)
        # flat_input_ids = input_tag_ids.view(-1, input_tag_ids.size(-1))
        embed = self.embed(flat_input_ids)
        # embed = self.dropout(embed)
        # print("embed", embed.size())
        input = embed.view(-1, num_aspect, flat_input_ids.size(1), self.hidden_size)
        # linear
        logit = self.fc(input)
        # print("logit", logit.size())
        return logit