import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.core.evaluator import Evaluator
from cogktr.utils.general_utils import init_cogktr

device,output_path = init_cogktr(
    device_id=0,
    output_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/",
    folder_tag="resume_training",
)

reader = QNLIReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
processor = QNLIProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

model = BaseSentencePairClassificationModel(plm="bert-base-cased", vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# evaluator = Evaluator(
#     model=model,
#     checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/train/2022-05-29-09:02:48/checkpoint-100",
#     dev_data=dev_dataset,
#     metrics=metric,
#     sampler=None,
#     drop_last=False,
#     collate_fn=None,
#     file_name="models.pt",
#     batch_size=32,
#     device=device,
#     user_tqdm=True,
# )
# evaluator.evaluate()
# print("End")

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
                  checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/simple_test1--2022-05-30--13-02-12.95/model/checkpoint-300",
                  validate_steps=100,      # validation setting
                  save_steps=100,         # when to save model result
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
