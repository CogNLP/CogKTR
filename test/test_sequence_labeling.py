import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *

torch.cuda.set_device(4)
device = torch.device('cuda:4')

reader = CONLL2003Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2003/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
processor = CONLL2003Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
train_dataset = processor.process(train_data)
dev_dataset = processor.process(dev_data)
test_dataset = processor.process(test_data)

model = BaseTextClassificationModel(plm="bert-base-cased", vocab=vocab)
metric = BaseTextClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
print("end")