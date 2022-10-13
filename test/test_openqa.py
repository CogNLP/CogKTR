import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.data.reader.naturalquestion_reader import NaturalQuestionsReader
from cogktr.data.processor.naturalquestions_processors.natural_questions_for_mrc_processor import NaturalQuestionsForMRCProcessor
from cogktr.core.metric.base_reading_comprehension_metric import BaseMRCMetric
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel

reader = NaturalQuestionsReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data",
    debug=True)
train_data, dev_data, test_data = reader.read_all()

enhancer = WorldEnhancer(knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph",
                         cache_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/enhanced_data",
                         cache_file="world_data",
                         reprocess=False,
                         load_entity_desc=False,
                         load_entity_embedding=False,
                         load_entity_kg=False,
                         load_retrieval_info=True)

top_k_relevant_psgs = 10
enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)
enhanced_dev_dict = enhancer.enhance_dev(dev_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)
enhanced_test_dict = enhancer.enhance_test(test_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)

processor = NaturalQuestionsForMRCProcessor(
    plm="bert-base-uncased",
    max_token_len=384,
)
train_dataset = processor.process_train(train_data, enhanced_train_dict)
dev_dataset = processor.process_dev(dev_data, enhanced_dev_dict)
test_dataset = processor.process_test(test_data, enhanced_test_dict)


model = BaseReadingComprehensionModel(plm="bert-base-uncased",vocab=None)
metric = BaseMRCMetric()
loss = nn.CrossEntropyLoss(ignore_index=384)



