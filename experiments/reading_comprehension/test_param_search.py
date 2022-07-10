import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.core.evaluator import Evaluator
from cogktr.utils.general_utils import init_cogktr,EarlyStopping
from cogktr.utils.log_utils import logger
import argparse
from argparse import Namespace
from cogktr.core.metric.base_reading_comprehension_metric import BaseMRCMetric
from cogktr.data.processor.squad2_processors.squad2_sembert_processor import Squad2SembertProcessor
from cogktr.models.old_sembert_model import BertForQuestionAnsertingTag
from cogktr.utils.general_utils import EarlyStopping
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel

parser = argparse.ArgumentParser("Global Config Argument Parser", allow_abbrev=False)
parser.add_argument("--lr", required=True, type=float, help='learning rate')
parser.add_argument("--device",required=True,type=int,help="device id")
parser.add_argument("--batch_size",default=32,type=int,help="training batch size")
# parser.add_argument("--resume", type=str, help='a specified logging path to resume training.\
#        It will fall back to run from initialization if no latest checkpoint are found.')
# parser.add_argument("--test", type=str, help='a specified logging path to test')
args = parser.parse_args()


device, output_path = init_cogktr(
    device_id=args.device,
    output_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/experimental_result/",
    # folder_tag="test",
    folder_tag="complex_metric_mrc_lr={}".format(args.lr,args.batch_size),
)

reader = Squad2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
train_data, dev_data, _ = reader.read_all()
vocab = reader.read_vocab()

processor = Squad2Processor(plm="bert-base-cased", max_token_len=512, vocab=vocab,debug=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)

model = BaseReadingComprehensionModel(plm="bert-base-cased",vocab=vocab)
metric = BaseMRCMetric()
loss = nn.CrossEntropyLoss(ignore_index=512)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(mode="max",patience=4,threshold=0.01,threshold_mode="abs",metric_name="EM")

logger.info("Lr:{}".format(
    [param_group["lr"] for param_group in optimizer.param_groups],
))

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=100,
                  batch_size=args.batch_size,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  early_stopping=early_stopping,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  # checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/simple_test1--2022-05-30--13-02-12.95/model/checkpoint-300",
                  validate_steps=500,  # validation setting
                  collate_fn=processor._collate,
                  save_steps=None,  # when to save model result
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
