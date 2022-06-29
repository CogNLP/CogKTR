import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.core.metric.base_reading_comprehension_metric import BaseMRCMetric
from cogktr.data.processor.squad2_processors.squad2_processor import Squad2Processor
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel

device,output_path = init_cogktr(
    device_id=4,
    output_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/experimental_result/",
    folder_tag="debug_mrc_metric",
)

reader = Squad2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
train_data, dev_data, _ = reader.read_all()
vocab = reader.read_vocab()

processor = Squad2Processor(plm="bert-base-cased", max_token_len=512, vocab=vocab,debug=True)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)

model = BaseReadingComprehensionModel(plm="bert-base-cased",vocab=vocab)
metric = BaseMRCMetric()
loss = nn.CrossEntropyLoss(ignore_index=512)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
early_stopping = EarlyStopping(mode="max",patience=8,threshold=0.01,threshold_mode="abs",metric_name="EM")

trainer = Trainer(model,
                  train_dataset,
                  dev_data=train_dataset,
                  n_epochs=200,
                  batch_size=16,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  early_stopping=early_stopping,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=50,
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  callbacks=None,
                  metric_key=None,
                  collate_fn=processor._collate,
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
print("end")





