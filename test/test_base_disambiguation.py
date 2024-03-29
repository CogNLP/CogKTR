import torch.nn as nn
import torch.optim as optim
from cogktr import init_cogktr, SemcorReader, SemcorProcessor
from cogktr import PlmAutoModel, BaseDisambiguationModel, BaseDisambiguationMetric, Trainer
from torch.utils.data import SequentialSampler

device, output_path = init_cogktr(
    device_id=6,
    output_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/experimental_result/",
    folder_tag="simple_test",
)

reader = SemcorReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
addition = reader.read_addition()

processor = SemcorProcessor(plm="bert-base-cased", max_token_len=512, vocab=vocab, addition=addition)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
dev_sampler = SequentialSampler(dev_dataset)

plm = PlmAutoModel(pretrained_model_name="bert-base-cased")
model = BaseDisambiguationModel(plm=plm, vocab=vocab)
metric = BaseDisambiguationMetric(segment_list=addition["dev"]["segmentation"])
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=5,
                  batch_size=25,
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
                  validate_steps=300,
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
