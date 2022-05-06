import torch
from cogktr import *

"""
reader=XXXReader()
train_data,valid_data,test_data=reader.read_all()

tagger=XXXTagger()
linker=XXXLinker()
searcher=XXXSearcher()
representer=XXXRepresenter()
enhancer=Enhancer([("tagger",tagger),("linker",linker),("searcher",searcher),("representer",representer)])
train_data=enhancer.enhance(train_data)
valid_data=enhancer.enhance(valid_data)
test_data=enhancer.enhance(test_data)

processor=XXXProcessor()
train_dataset=processor.process(train_data)
valid_dataset=processor.process(valid_data)
test_dataset=processor.process(test_data)

model=XXXModel()
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



