import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr

device, output_path = init_cogktr(
    device_id=5,
    output_path="/data/mentianyi/CogKTR/datapath/text_classification/SST_2/experimental_result",
    folder_tag="simple_test",
)  # TODO:device_id这里限制不住

reader = SST2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = Enhancer(return_entity_ebd=True)
enhancer.set_config(
    WikipediaSearcherPath="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia/raw_data/entity.jsonl",
    WikipediaEmbedderPath="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl")
processor = SST24KGEMBProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab, enhancer=enhancer)
train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)#TODO:back
# test_dataset = processor.process_test(test_data)#TODO:back

model = KgEmbModel4TC(plm="bert-base-cased", vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=train_dataset,  # TODO:dev_dataset
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
                  validate_steps=20,
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
                  dev_collate_fn=train_dataset.to_dict,
                  )
trainer.train()
print("end")
