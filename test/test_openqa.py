import torch.nn as nn
import torch.optim as optim
import torch
from collections import OrderedDict
from cogktr import *
from cogktr.data.reader.naturalquestion_reader import NaturalQuestionsReader
from cogktr.data.processor.naturalquestions_processors.natural_questions_for_mrc_processor import NaturalQuestionsForMRCProcessor
from cogktr.core.metric.base_openqa_metric import BaseOpenQAMetric
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel

device = torch.device("cuda:1")

reader = NaturalQuestionsReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data",
    debug=True)
train_data, dev_data, test_data = reader.read_all()

enhancer = WorldEnhancer(knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph",
                         cache_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/enhanced_data",
                         cache_file="world_data",
                         reprocess=True,
                         load_entity_desc=False,
                         load_entity_embedding=False,
                         load_entity_kg=False,
                         load_retrieval_info=True)

top_k_relevant_psgs = 20
# enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)
enhanced_dev_dict = enhancer.enhance_dev(dev_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)
# enhanced_test_dict = enhancer.enhance_test(test_data,enhanced_key_1='question',return_top_k_relevent_psgs=top_k_relevant_psgs)

processor = NaturalQuestionsForMRCProcessor(
    plm="bert-base-uncased",
    max_token_len=384,
)
# train_dataset = processor.process_train(train_data, enhanced_train_dict)
dev_dataset = processor.process_dev(dev_data, enhanced_dev_dict)
# test_dataset = processor.process_test(test_data, enhanced_test_dict)


model = BaseReadingComprehensionModel(plm="bert-base-uncased",vocab=None)
metric = BaseOpenQAMetric()

evaluator = Evaluator(
    model=model,
    checkpoint_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data/",
    dev_data=dev_dataset,
    metrics=metric,
    sampler=None,
    drop_last=False,
    collate_fn=processor._collate,
    file_name="bert-base-mrc-openqa.pt",
    batch_size=32,
    device=device,
    user_tqdm=True,
)
evaluator.evaluate()
print("End")


# # Convert the original model weight file to adapt to current model
# model_file = '/data/hongbang/projects/DPR/downloads/checkpoint/reader/nq-single/hf-bert-base.cp'
# state_dict = torch.load(model_file)
# model_state_dict = OrderedDict()
# for key,value in state_dict["model_dict"].items():
#     if not key.startswith("qa_classifier"):
#         key_items = key.split(".")
#         if key_items[0] == 'encoder':
#             key_items[0] = 'bert'
#         elif key_items[0] == 'qa_outputs':
#             key_items[0] = 'linear'
#         model_state_dict.update({".".join(key_items): value})
#
# model = BaseReadingComprehensionModel(plm='bert-base-uncased',vocab=None)
# model.to(torch.device('cuda:0'))
# model.load_state_dict(model_state_dict,strict=False)
# print("Done!")
# model.to(torch.device('cpu'))
# torch.save(model.state_dict(),'/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data/bert-base-mrc-openqa.pt')

