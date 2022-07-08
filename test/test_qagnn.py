import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr

device, output_path = init_cogktr(
    device_id=2,
    output_path="/data/mentianyi/code/CogKTR/datapath/question_answering/CommonsenseQA_for_QAGNN/experimental_result",
    folder_tag="simple_test",
)

reader = CommonsenseqaQagnnReader(
    raw_data_path="/data/mentianyi/code/CogKTR/datapath/question_answering/CommonsenseQA_for_QAGNN/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
addition = reader.read_addition()

processor = CommonsenseqaQagnnProcessor(plm="roberta-large", max_token_len=100, vocab=vocab, addition=addition)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name="roberta-large")
model = QAGNNModel(plm=plm, vocab=vocab, addition=addition)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
grouped_parameters = [
    {'params': [p for n, p in model.plm._plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 1e-05},
    {'params': [p for n, p in model.plm._plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 1e-05},
    {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 0.001},
    {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 0.001},
]
optimizer = optim.RAdam(grouped_parameters)
# optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=5,
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
                  collate_fn=train_dataset.to_dict,
                  dev_collate_fn=dev_dataset.to_dict,
                  )
trainer.train()
print("end")
