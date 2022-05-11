import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *

torch.cuda.set_device(4)
device = torch.device('cuda:4')

# STSB dataset
# reader = STSBReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/STS_B/raw_data")
# train_data, dev_data, test_data = reader.read_all()
# vocab = reader.read_vocab()
# processor = STSBProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
# train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)
# test_dataset = processor.process_test(test_data)
# print("end")


# QNLI dataset
reader = QNLIReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
processor = QNLIProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
train_dataset = processor.process_train(dev_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

model = BaseSentencePairModel(plm="bert-base-cased", vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
