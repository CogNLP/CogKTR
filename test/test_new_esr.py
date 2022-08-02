import os

os.environ['CUDA_VISIBLE_DEVICES'] = "8"

import torch.nn as nn
import torch.optim as optim
from cogktr import init_cogktr, SemcorReader, LinguisticsEnhancer, TSemcorProcessor
from cogktr import PlmAutoModel, BaseDisambiguationModel, BaseDisambiguationMetric, Trainer
from torch.utils.data import SequentialSampler

device, output_path = init_cogktr(
    device_id=0,
    output_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/experimental_result/",
    folder_tag="simple_test",
)

reader = SemcorReader(
    raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
addition = reader.read_addition()

enhancer = LinguisticsEnhancer(load_wordnet=True,
                               cache_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/enhanced_data",
                               cache_file="linguistics_data",
                               reprocess=False)
enhanced_train_dict = enhancer.enhance_train(datable=train_data,
                                             return_wordnet=True,
                                             enhanced_key_1="instance_list",
                                             pos_key="instance_pos_list")
enhanced_dev_dict = enhancer.enhance_dev(datable=dev_data,
                                         return_wordnet=True,
                                         enhanced_key_1="instance_list",
                                         pos_key="instance_pos_list")

processor = TSemcorProcessor(plm="bert-base-cased", max_token_len=512, vocab=vocab, addition=addition)
train_dataset = processor.process_train(train_data, enhanced_dict=enhanced_train_dict)
dev_dataset = processor.process_dev(dev_data, enhanced_dict=enhanced_dev_dict)
dev_sampler = SequentialSampler(dev_dataset)

tmodel= PlmAutoModel(pretrained_model_name="bert-base-cased")
kmodel = BaseDisambiguationModel(plm=tmodel, vocab=vocab)
metric = BaseDisambiguationMetric(segment_list=addition["dev"]["segmentation"])
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(kmodel.parameters(), lr=0.00001)

trainer = Trainer(kmodel,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
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
