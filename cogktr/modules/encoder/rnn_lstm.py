import torch.nn as nn
import torch


class RNN_LSTM(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 bias=True,
                 batch_first=True,
                 dropout=0):
        super().__init__()
        if bidirectional:
            hidden_size = int(hidden_size / 2)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, input_embs=None):
        x, (h, c) = self.lstm(input_embs)
        return x


if __name__ == "__main__":
    lstm = RNN_LSTM(embedding_dim=100,
                    hidden_size=100,
                    num_layers=1,
                    bidirectional=True)
    input_embs = torch.zeros((20, 128, 100))  # (Batch,Len,Dim)
    output_input_embs = lstm(input_embs=input_embs)  # (Batch,Len,Dim)
    print("end")
