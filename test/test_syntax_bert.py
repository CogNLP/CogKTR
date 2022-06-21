import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr

device, output_path = init_cogktr(
    device_id=7,
    output_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/experimental_result/",
    folder_tag="test_dict_input",
)

reader = QnliReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = Enhancer(return_syntax=True,
                    reprocess=True,
                    save_file_name="pre_enhanced_data",
                    datapath="/data/mentianyi/code/CogKTR/datapath",
                    enhanced_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/enhanced_data")
# enhanced_train_dict = enhancer.enhance_train(train_data, enhanced_key_1="sentence", enhanced_key_2="question")
enhanced_dev_dict = enhancer.enhance_dev(dev_data, enhanced_key_1="sentence", enhanced_key_2="question")
# enhanced_test_dict = enhancer.enhance_test(test_data, enhanced_key_1="sentence", enhanced_key_2="question")

processor = QnliForSyntaxBertProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
train_dataset = processor.process_train(dev_data, enhanced_dict=enhanced_dev_dict)
dev_dataset = processor.process_dev(dev_data, enhanced_dict=enhanced_dev_dict)
# test_dataset = processor.process_test(test_data)

model = BaseSentencePairClassificationModel(plm="bert-base-cased", vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=32,
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
