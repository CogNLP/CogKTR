import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.enhancers.new_enhancer import NewEnhancer

# device, output_path = init_cogktr(
#     device_id=8,
#     output_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/",
#     folder_tag="enhance_commonsense_safe_debug",
# )

reader = OpenBookQAReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = NewEnhancer(config_path="/data/hongbang/CogKTR/cogktr/utils/config/enhancer_config.json",
                       knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph",
                       save_file_name="pre_enhanced_data",
                       enhanced_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/enhanced_data",
                       load_conceptnet=True,
                       reprocess=True)

enhanced_sentence_dict = enhancer.enhance_sentence(sentence="Bert likes reading in the Sesame Street Library.",
                                                   return_conceptnet=True)






