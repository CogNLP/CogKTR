import torch
from cogktr import *

"""
tagger=XXXTagger()
linker=XXXLinker()
searcher=XXXSearcher()
embedder=XXXEmbedder()
enhancer=Enhancer([("tagger",tagger),("linker",linker),("searcher",searcher),("embedder",embedder)])

reader=XXXReader()
train_data=reader.read_train_data()
valid_data=reader.read_valid_data()
test_data=reader.read_test_data()

train_data=enhancer.enhance_data(train_data)
valid_data=enhancer.enhance_data(valid_data)
test_data=enhancer.enhance_data(test_data)

processor=XXXProcessor()
train_dataset=processor.process(train_data)
valid_dataset=processor.process(valid_data)
test_dataset=processor.process(test_data)

model=XXXModel()
model=enhancer.enhance_model(model)
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



