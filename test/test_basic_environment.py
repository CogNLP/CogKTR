import numpy as np
import torch
import transformers
from transformers import BertConfig
from cogktr.utils.log_utils import logger,init_logger

print("Hello World!")

init_logger()
logger.info("Hello World!")
init_logger("log.txt")
logger.info("Hello World Again!")


"""
tagger=XXXTagger()
linker=XXXLinker()
searcher=XXXSearcher()
embedder=XXXEmbedder()
enhancers=Enhancer([("tagger",tagger),("linker",linker),("searcher",searcher),("embedder",embedder)])

reader=XXXReader()
train_data=reader.read_train_data()
valid_data=reader.read_valid_data()
test_data=reader.read_test_data()
train_data=enhancers.enhance_data(train_data)
valid_data=enhancers.enhance_data(valid_data)
test_data=enhancers.enhance_data(test_data)

processor=XXXProcessor()
train_dataset=processor.process(train_data)
valid_dataset=processor.process(valid_data)
test_dataset=processor.process(test_data)

model=XXXModel()
model=enhancers.enhance_model(model)
loss=XXXLoss()
metric=XXXMetric()
optimizer=XXXOptimizer()

trainer=Trainer()
trainer.train()

evaluator=Evaluator()
evaluator=evaluator.evaluate()

predictor=Predictor()
predictor=predictor.predicte()
"""
