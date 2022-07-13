import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from torch.utils.data import SequentialSampler

device, output_path = init_cogktr(
    device_id=2,
    output_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/experimental_result/",
    folder_tag="simple_test",
)

reader = SemcorReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = SemcorProcessor(plm="roberta-large", max_token_len=256, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)
dev_sampler = SequentialSampler(dev_dataset)

plm = PlmAutoModel(pretrained_model_name="roberta-large")
model = EsrModel(plm=plm, vocab=vocab, segment_list=reader.segment_list)
metric = BaseDisambiguationMetric(segment_list=reader.segment_list)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

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
                  dev_sampler=dev_sampler,
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
                  )
trainer.train()
print("end")
