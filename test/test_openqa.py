import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.data.reader.naturalquestion_reader import NaturalQuestionsReader

reader = NaturalQuestionsReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data")
train_data, dev_data, test_data = reader.read_all()

enhancer = WorldEnhancer(knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph",
                         cache_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/enhanced_data",
                         cache_file="world_data",
                         reprocess=True,
                         load_entity_desc=False,
                         load_entity_embedding=False,
                         load_entity_kg=False,
                         load_retrieval_info=True)

top_k_relevant_psgs = 100
enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)
enhanced_dev_dict = enhancer.enhance_dev(dev_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)
enhanced_test_dict = enhancer.enhance_test(test_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)


print("Done!")




