import torch.nn as nn
import torch.optim as optim
from cogktr import init_cogktr, MultisegchnsentibertReader, MultisegchnsentibertProcessor
from cogktr import PlmAutoModel, BaseTextClassificationModel, BaseClassificationMetric, Trainer

device, output_path = init_cogktr(
    device_id=9,
    output_path="/data/mentianyi/code/CogKTR/datapath/text_classification/MultiSegChnSentiBERT/experimental_result",
    folder_tag="simple_test",
    seed=1,
)

reader = MultisegchnsentibertReader(
    raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/MultiSegChnSentiBERT/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = MultisegchnsentibertProcessor(plm="bert-base-chinese", max_token_len=128, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name="bert-base-chinese")
model = BaseTextClassificationModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
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
                  )
trainer.train()
print("end")
