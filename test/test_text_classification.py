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
import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *

torch.cuda.set_device(4)
device = torch.device('cuda:4')

reader = SST2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = SST2Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

model = BaseTextClassificationModel(plm="bert-base-cased", vocab=vocab)
metric = BaseTextClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# TODO: Set the path as data/task/datasetname/modelname-time-epoch-filenamedefinebuyourself/checkpoint
trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=50,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  save_path=None,
                  save_file=None,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=None,
                  save_steps=None,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  device_ids=[4],
                  callbacks=None,
                  metric_key=None,
                  writer_path=None,
                  fp16=False,
                  fp16_opt_level='O1',
                  logger_path=None)
trainer.train()
print("end")
