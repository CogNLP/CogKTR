import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.core.evaluator import Evaluator
from cogktr.utils.general_utils import init_cogktr,EarlyStopping
from cogktr.utils.log_utils import logger
import argparse

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
    output_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/",
    folder_tag="param_search_lr={}_batchsize={}".format(args.lr,args.batch_size),
)

reader = QnliReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()
processor = QnliProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab, debug=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

model = BaseSentencePairClassificationModel(plm="bert-base-cased", vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(mode="max",patience=2,threshold=0.05,threshold_mode="rel",metric_name="F1")

logger.info("Lr:{}".format(
    [param_group["lr"] for param_group in optimizer.param_groups],
))

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=100,
                  batch_size=32,
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
