import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr

device, output_path = init_cogktr(
    device_id=3,
    output_path="/data/mentianyi/code/CogKTR/datapath/text_classification/MultiSegChnSentiBERT/experimental_result",
    folder_tag="simple_test",
)

reader = MultisegchnsentibertReader(
    raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/MultiSegChnSentiBERT/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = MultisegchnsentibertProcessor(max_token_len=128, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

model = HLGModel(plm="bert-base-cased", vocab=vocab, hidden_size=768, hidden_dropout_prob=0.1)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

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
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=100,
                  save_steps=100,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  callbacks=None,
                  metric_key=None,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=processor.train_collate,
                  dev_collate_fn=processor.dev_collate,
                  )
trainer.train()
print("end")
