import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from torch.utils.data import SequentialSampler

device, output_path = init_cogktr(
    device_id=2,
    output_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/experimental_result/",
    folder_tag="simple_test",
)

reader = TSemcorReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
addition = reader.read_addition()


print("end")
