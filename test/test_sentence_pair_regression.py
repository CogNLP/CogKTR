import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *

torch.cuda.set_device(4)
device = torch.device('cuda:4')

reader = STSBReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/STS_B/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
processor = STSBProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)
print("end")
