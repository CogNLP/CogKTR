import torch.nn as nn

class CNN_conv1d(nn.Module):
    def __init__(self, config, filter_size):
        super(CNN_conv1d, self).__init__()
        self.char_dim = config.hidden_size
        self.filter_size = filter_size #max_word_length
        self.out_channels = self.char_dim
        self.char_cnn =nn.Conv1d(self.char_dim, self.char_dim,kernel_size=self.filter_size,
                     padding=0)
        self.relu = nn.ReLU()
        #print("dropout:",str(config.hidden_dropout_prob))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs, max_word_len):
        """
        Arguments:
            inputs: [batch_size, word_len, char_len]
        """
        if(len(inputs.size())>3):
            bsz, word_len,  max_word_len, dim = inputs.size()
            #print(bsz, word_len,  max_word_len, dim)
        else:
            bsz, word_len, dim = inputs.size()
            word_len = int(word_len / max_word_len)

        inputs = inputs.view(-1, max_word_len, dim)
        x = inputs.transpose(1, 2)
        x = self.char_cnn(x)
        x = self.relu(x)
        x = F.max_pool1d(x, kernel_size=x.size(-1))
        x = self.dropout(x.squeeze())

        return x.view(bsz, word_len, -1)
