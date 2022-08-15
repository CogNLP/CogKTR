import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.data.processor.openbookqa_processors.openbookqa_processor import OpenBookQAProcessor

device, output_path = init_cogktr(
    device_id=9,
    output_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/",
    folder_tag="test_obqa_roberta_large",
)

reader = OpenBookQAReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = OpenBookQAProcessor(
    plm="roberta-large",
    max_token_len=100,
    vocab=vocab,
    debug=False,
)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_train(dev_data)

plm = PlmAutoModel(pretrained_model_name="roberta-large")
model = BaseQuestionAnsweringModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)
early_stopping = EarlyStopping(mode="max",patience=3,threshold=0.001,threshold_mode="abs",metric_name="Acc")

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=10,
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
                  validate_steps=None,
                  early_stopping=early_stopping,
                  save_by_metric="Acc",
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  callbacks=None,
                  metric_key=None,
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
print("end")
