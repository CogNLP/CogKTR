import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr

device, output_path = init_cogktr(
    device_id=5,
    output_path="/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2005_srl_subset/experimental_result",
    folder_tag="simple_test",
)

reader = Conll2005SrlSubsetReader(
    raw_data_path="/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2005_srl_subset/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

print("end")
