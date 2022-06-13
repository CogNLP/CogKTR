import torch.nn as nn
import torch


class CNN_TokenCNN(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size=3,
                 padding=1,
                 act_fun="ReLU"):
        super().__init__()
        self.conv = nn.Conv1d(input_size,
                              out_channels=hidden_size,
                              kernel_size=(kernel_size,),
                              padding=padding)
        self.act = activate_function_dict[act_fun]

    def forward(self, input_embs):
        x = input_embs.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        x = x.transpose(1, 2)
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
    textcnn = TokenCNN(input_size=100, hidden_size=100)
    input_embs = torch.zeros((20, 128, 100))  # (Batch,Len,Dim)
    output_input_embs = textcnn(input_embs=input_embs)  # (Batch,Len,Dim)
    print("end")
