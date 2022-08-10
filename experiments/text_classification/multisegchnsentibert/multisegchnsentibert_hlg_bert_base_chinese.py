import torch.nn as nn
import torch.optim as optim
from cogktr import init_cogktr, MultisegchnsentibertReader, MultisegchnsentibertForHLGProcessor
from cogktr import PlmAutoModel, HLGModel, BaseClassificationMetric, Trainer

device, output_path = init_cogktr(
    device_id=7,
    output_path="/data/mentianyi/code/CogKTR/datapath/text_classification/MultiSegChnSentiBERT/experimental_result",
    folder_tag="simple_test",
    seed=1,
)

reader = MultisegchnsentibertReader(
    raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/MultiSegChnSentiBERT/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = MultisegchnsentibertForHLGProcessor(max_token_len=128, vocab=vocab, plm="bert-base-chinese")
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name="bert-base-chinese")
kmodel = HLGModel(plm=plm, vocab=vocab, hidden_dropout_prob=0.1)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(kmodel.parameters(), lr=0.00001)

trainer = Trainer(kmodel,
                  train_dataset,
                  dev_data=test_dataset,
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
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=processor.train_collate,
                  dev_collate_fn=processor.test_collate,
                  )
trainer.train()
print("end")
