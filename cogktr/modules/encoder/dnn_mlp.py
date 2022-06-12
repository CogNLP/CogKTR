import torch.nn as nn
import torch


class DNN_MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_1_dim,
                 act_1_fun,
                 out_features,
                 hidden_2_dim=None,
                 act_2_fun="ReLU",
                 hidden_3_dim=None,
                 act_3_fun="ReLU",
                 ):
        super().__init__()
        self.in_features = in_features
        self.hidden_1_dim = hidden_1_dim
        self.act_1_fun = act_1_fun
        self.out_features = out_features
        self.hidden_2_dim = hidden_2_dim
        self.act_2_fun = act_2_fun
        self.hidden_3_dim = hidden_3_dim
        self.act_3_fun = act_3_fun

        self.hidden_1 = nn.Linear(in_features, hidden_1_dim)
        self.act_1 = activate_function_dict[act_1_fun]
        self.hidden_out = nn.Linear(hidden_1_dim, out_features)
        if hidden_2_dim is not None:
            self.hidden_2 = nn.Linear(hidden_1_dim, hidden_2_dim)
            self.act_2 = activate_function_dict[act_2_fun]
            self.hidden_out = nn.Linear(hidden_2_dim, out_features)
        if hidden_2_dim is not None and hidden_3_dim is not None:
            self.hidden_3 = nn.Linear(hidden_2_dim, hidden_3_dim)
            self.act_3 = activate_function_dict[act_3_fun]
            self.hidden_out = nn.Linear(hidden_3_dim, out_features)

    def forward(self, input_embs=None):
        x = self.hidden_1(input_embs)
        x = self.act_1(x)
        if self.hidden_2_dim is None:
            x = self.hidden_out(x)
        if self.hidden_2_dim is not None:
            x = self.hidden_2(x)
            x = self.act_2(x)
            x = self.hidden_out(x)
        if self.hidden_2_dim is not None and self.hidden_3_dim is not None:
            x = self.hidden_2(x)
            x = self.act_2(x)
            x = self.hidden_3(x)
            x = self.act_3(x)
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
                  out_features=6,
                  hidden_1_dim=100,
                  act_1_fun="ReLU",
                  hidden_2_dim=200,
                  act_2_fun="ReLU", )
    input_embs = torch.zeros((20, 100))  # (Batch,Dim)
    output_input_embs = mlp(input_embs=input_embs)  # (Batch,Dim)
    print("end")
