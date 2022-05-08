import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *

torch.cuda.set_device(4)
device = torch.device('cuda:4')

reader = SQUAD2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
print("end")
