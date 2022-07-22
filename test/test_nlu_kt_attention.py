import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr

# initiate
device, output_path = init_cogktr(
    device_id=4,
    output_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_5/experimental_result",
    folder_tag="simple_test",
)

reader = Sst5Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_5/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = WorldEnhancer(knowledge_graph_path="/data/mentianyi/code/CogKTR/datapath/knowledge_graph",
                         cache_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_5/enhanced_data",
                         cache_file="world_data",
                         reprocess=False,
                         load_entity_desc=True,
                         load_entity_embedding=False,
                         load_entity_kg=False)
enhanced_train_dict = enhancer.enhance_train(datable=train_data,
                                             enhanced_key_1="sentence",
                                             return_entity_desc=True,
                                             return_entity_embedding=False,
                                             return_entity_kg=False)
enhanced_dev_dict = enhancer.enhance_dev(datable=dev_data,
                                         enhanced_key_1="sentence",
                                         return_entity_desc=True,
                                         return_entity_embedding=False,
                                         return_entity_kg=False)
enhanced_test_dict = enhancer.enhance_test(datable=test_data,
                                           enhanced_key_1="sentence",
                                           return_entity_desc=True,
                                           return_entity_embedding=False,
                                           return_entity_kg=False)
# processor = Sst5Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
# train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)
# test_dataset = processor.process_test(test_data)
processor = Sst5ForKtattProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
train_dataset = processor.process_train(data=train_data, enhanced_dict=enhanced_train_dict)
dev_dataset = processor.process_dev(data=dev_data, enhanced_dict=enhanced_dev_dict)
test_dataset = processor.process_test(data=test_data, enhanced_dict=enhanced_test_dict)

plm = PlmBertModel(pretrained_model_name="bert-base-cased")
model = BaseTextClassificationModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=2000,
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
                  save_steps=1000,
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
