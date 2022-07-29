import torch.nn as nn
import torch.optim as optim
from cogktr import *

device, output_path = init_cogktr(
    device_id=7,
    # output_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/experimental_result",  # SST_2
    output_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_5/experimental_result",  # SST_5
    folder_tag="test_print_metric",
)

# reader = Sst2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")  # SST_2
reader = Sst5Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_5/raw_data")  # SST_5
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

# processor = Sst2Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)  # SST_2
processor = Sst5Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)  # SST_5
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name="bert-base-cased")
model = BaseTextClassificationModel(plm=plm, vocab=vocab)
# metric = BaseClassificationMetric(mode="binary")  # SST_2
metric = BaseClassificationMetric(mode="multi")  # SST_5
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=100,
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
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
print("end")
