import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.core.evaluator import Evaluator
from cogktr.utils.general_utils import init_cogktr
# from cogktr.models.sembert_model import BertForSequenceClassificationTag
from cogktr.models.old_sembert_model import BertForSequenceClassificationTag
from argparse import Namespace
from cogktr.models.sembert_model import SembertForSequenceClassification
from cogktr.modules.encoder.sembert import SembertEncoder

device, output_path = init_cogktr(
    device_id=7,
    output_path="/data/hongbang/CogKTR/datapath/text_classification/SST_2/experimental_result/",
    folder_tag="sembert_with_tag",
)

reader = Sst2Reader(raw_data_path="/data/hongbang/CogKTR/datapath/text_classification/SST_2/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = Enhancer(reprocess=False,
                    return_srl=True,
                    save_file_name="pre_enhanced_data",
                    datapath="/data/hongbang/CogKTR/datapath",
                    enhanced_data_path="/data/hongbang/CogKTR/datapath/text_classification/SST_2/enhanced_data")
enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key_1="sentence")
enhanced_dev_dict = enhancer.enhance_dev(dev_data,enhanced_key_1="sentence")
enhanced_test_dict = enhancer.enhance_test(test_data,enhanced_key_1="sentence")


processor = Sst2SembertProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab,debug=False)
train_dataset = processor.process_train(train_data,enhanced_train_dict)
dev_dataset = processor.process_dev(dev_data,enhanced_dev_dict)

tag_config = {
   "tag_vocab_size":len(vocab["tag_vocab"]),
   "hidden_size":10,
   "output_dim":10,
   "dropout_prob":0.1,
   "num_aspect":3
}
# tag_config = Namespace(**tag_config)

plm = SembertEncoder.from_pretrained("bert-base-cased",tag_config=tag_config)
# plm = SembertEncoder.from_pretrained("bert-large-uncased",tag_config=None)
model = SembertForSequenceClassification(
    vocab=vocab,
    plm=plm,
)

metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
early_stopping = EarlyStopping(mode="max",patience=3,threshold=0.0001,threshold_mode="abs",metric_name="F1")

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
                  # checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/simple_test1--2022-05-30--13-02-12.95/model/checkpoint-300",
                  validate_steps=100,  # validation setting
                  save_steps=None,  # when to save model result
                  early_stopping=early_stopping,
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

# evaluator = Evaluator(
#     model=model,
#     checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/simple_test1--2022-05-30--13-02-12.95/model/checkpoint-400",
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
