import torch.nn as nn
import torch


class DNN_MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_dim,
                 out_features,
                 act_fun="ReLU",
                 ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.act_fun = act_fun
        self.out_features = out_features

        self.hidden_in = nn.Linear(in_features, hidden_dim)
        self.act = activate_function_dict[act_fun]
        self.hidden_out = nn.Linear(hidden_dim, out_features)

    def forward(self, input_embs=None):
        x = self.hidden_in(input_embs)
        x = self.act(x)
        x = self.hidden_out(x)
        return x


activate_function_dict = {"ReLU": nn.ReLU(),
                          "RReLU": nn.RReLU(),
                          "LeakyReLU": nn.LeakyReLU(),
                          "PReLU": nn.PReLU(),
                          "Sofplus": nn.Softplus(),
                          "ELU": nn.ELU(),
                          "CELU": nn.CELU(),
                          "SELU": nn.SELU(),
                          "GELU": nn.GELU(),
                          "ReLU6": nn.ReLU6(),
                          "Sigmoid": nn.Sigmoid(),
                          "Tanh": nn.Tanh(),
                          "Softsign": nn.Softsign(),
                          "Hardtanh": nn.Hardtanh(),
                          "Tanhshrink": nn.Tanhshrink(),
                          "Softshrink": nn.Softshrink(),
                          "Hardshrink": nn.Hardshrink(),
                          "LogSigmoid": nn.LogSigmoid(),
                          "Softmin": nn.Softmin(),
                          "Softmax": nn.Softmax(),
                          "LogSoftmax": nn.LogSoftmax()}

if __name__ == "__main__":
    mlp = DNN_MLP(in_features=100,
                  hidden_dim=200,
                  out_features=6,
                  act_fun="ReLU")
    input_embs = torch.zeros((20, 100))  # (Batch,Dim)
    output_input_embs = mlp(input_embs=input_embs)  # (Batch,Dim)
    print("end")
