import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr

# device, output_path = init_cogktr(
#     device_id=8,
#     output_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/",
#     folder_tag="enhance_commonsense_safe_debug",
# )
reader = OpenBookQAReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()






