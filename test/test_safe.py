# import torch.nn as nn
# import torch.optim as optim
# from cogktr import *
# from cogktr.utils.general_utils import init_cogktr
# from cogktr.enhancers.new_enhancer import NewEnhancer
#
# # device, output_path = init_cogktr(
# #     device_id=8,
# #     output_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/",
# #     folder_tag="enhance_commonsense_safe_debug",
# # )
#
# reader = OpenBookQAReader(
#     raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
# train_data, dev_data, test_data = reader.read_all()
# vocab = reader.read_vocab()
#
# enhancer = NewEnhancer(config_path="/data/hongbang/CogKTR/cogktr/utils/config/enhancer_config.json",
#                        knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph",
#                        save_file_name="pre_enhanced_data",
#                        enhanced_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/enhanced_data",
#                        load_conceptnet=True,
#                        reprocess=True)
#
# enhanced_sentence_dict = enhancer.enhance_sentence(sentence="Bert likes reading in the Sesame Street Library.",
#                                                    return_conceptnet=True)
#
#
#
#
#
#

import json
original_matcher_pattern_file = "/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet/original_concept/matcher_patterns.json"
my_matcher_pattern_file = "/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet/matcher_patterns.json"

with open(original_matcher_pattern_file, "r", encoding="utf8") as fin:
    original_matcher_pattern = json.load(fin)

with open(my_matcher_pattern_file, "r", encoding="utf8") as fin:
    my_matcher_pattern= json.load(fin)

print("Debug Usage")

outliers = []
for (original_key,original_value) in original_matcher_pattern.items():
    if original_key not in my_matcher_pattern:
        outliers.append(original_key)