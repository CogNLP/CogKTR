import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.io_utils import save_pickle,load_pickle
from cogktr.utils.general_utils import init_cogktr
from cogktr.data.processor.openbookqa_processors.openbookqa_for_safe_processor import OpenBookQAForSafeProcessor
from cogktr.enhancers.conceptnet_enhancer import ConceptNetEnhancer

# device, output_path = init_cogktr(
#     device_id=8,
#     output_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/",
#     folder_tag="enhance_commonsense_safe_debug",
# )

reader = OpenBookQAReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = ConceptNetEnhancer(
        knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet",
        cache_path='/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/enhanced_data',
        reprocess=True
    )

# enhanced_train_dict = enhancer.enhance_train(train_data, enhanced_key="statement", enhanced_key_pair="answer_text")
# enhanced_dev_dict = enhancer.enhance_dev(train_data, enhanced_key="statement", enhanced_key_pair="answer_text")
# enhanced_test_dict = enhancer.enhance_test(train_data, enhanced_key="statement", enhanced_key_pair="answer_text")
# _,enhanced_dev_dict,enhacned_test_dict = enhancer.enhance_all(vocab)
enhanced_train_dict = load_pickle("/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/enhanced_data/conceptnet_enhanced_cache_train_with_metapath.pkl")
meta_path_set = load_pickle("/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/enhanced_data/meta_path_subset.pkl")
cpnet_vocab = Vocabulary()
cpnet_vocab.add_sequence(list(meta_path_set))
cpnet_vocab.create()
vocab["metapath"] = cpnet_vocab
processor = OpenBookQAForSafeProcessor(
    plm="bert-base-cased",
    max_token_len=256,
    vocab=vocab
)
train_dataset = processor.process_train(train_data,enhanced_train_dict)





# import json
# original_matcher_pattern_file = "/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet/original_concept/matcher_patterns.json"
# my_matcher_pattern_file = "/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet/matcher_patterns.json"
#
# with open(original_matcher_pattern_file, "r", encoding="utf8") as fin:
#     original_matcher_pattern = json.load(fin)
#
# with open(my_matcher_pattern_file, "r", encoding="utf8") as fin:
#     my_matcher_pattern= json.load(fin)
#
# print("Debug Usage")
#
# outliers = []
# for (original_key,original_value) in original_matcher_pattern.items():
#     if original_key not in my_matcher_pattern:
#         outliers.append(original_key)