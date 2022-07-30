import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.core.metric.base_reading_comprehension_metric import BaseMRCMetric
from cogktr.data.processor.squad2_processors.squad2_processor import Squad2Processor
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel


enhancer = LinguisticsEnhancer(load_ner=False,
                               load_srl=False,
                               load_syntax=True,
                               load_wordnet=False,
                               cache_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/enhanced_data",
                               cache_file="syntax_data",
                               reprocess=True)

reader = Squad2Reader(raw_data_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
train_data, dev_data, _ = reader.read_all()
vocab = reader.read_vocab()

# enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key_1="question_text",enhanced_key_2="doc_tokens",return_syntax=True)
enhanced_dev_dict = enhancer.enhance_dev(dev_data,enhanced_key_1="question_text",enhanced_key_2="doc_tokens",return_syntax=True)

# processor = Squad2Processor(plm="bert-base-cased", max_token_len=512, vocab=vocab,debug=False)
# train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)



