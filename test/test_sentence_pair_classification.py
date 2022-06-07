# CUDA_VISIBLE_DEVICES="2,3,4,5"  python -m torch.distributed.launch --nproc_per_node 4 test_sentence_pair_classification.py

import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn as nn

from cogktr import *
from cogktr.utils.general_utils import init_cogktr

def main(local_rank):
    device,output_path = init_cogktr(
        output_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/",
        folder_tag="test_dict_input",
        rank=local_rank,
        seed=1 + local_rank,
    )
    if local_rank != 0:
        dist.barrier()
    reader = QnliReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    if local_rank == 0:
        dist.barrier()

    processor = QnliProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)

    train_sampler = DistributedSampler(train_dataset)
    dev_sampler = DistributedSampler(dev_dataset)
    test_sampler = DistributedSampler(test_dataset)

    model = BaseSentencePairClassificationModel(plm="bert-base-cased", vocab=vocab)
    metric = BaseClassificationMetric(mode="binary")
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    dist.barrier()

    trainer = Trainer(model,
                      train_dataset,
                      dev_data=dev_dataset,
                      n_epochs=100,
                      batch_size=32,
                      loss=loss,
                      optimizer=optimizer,
                      scheduler=None,
                      metrics=metric,
                      train_sampler=train_sampler,
                      dev_sampler=dev_sampler,
                      drop_last=False,
                      gradient_accumulation_steps=1,
                      num_workers=5,
                      print_every=None,
                      scheduler_steps=None,
                      validate_steps=None,      # validation setting
                      save_steps=None,         # when to save model result
                      checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/test_dict_input--2022-06-07--05-35-30.21/model/checkpoint-819",
                      output_path=output_path,
                      grad_norm=1,
                      use_tqdm=True,
                      device=device,
                      callbacks=None,
                      metric_key=None,
                      fp16=False,
                      fp16_opt_level='O1',
                      rank=local_rank,
                      )
    trainer.train()
    print("end")




def spmd_main(local_rank):
    dist.init_process_group(backend="nccl")
    main(local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    spmd_main(local_rank)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from cogktr import *
# from cogktr.core.evaluator import Evaluator
# from cogktr.utils.general_utils import init_cogktr
#
# device,output_path = init_cogktr(
#     device_id=6,
#     output_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/",
#     folder_tag="test_dict_input",
# )
#
# reader = QnliReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
# train_data, dev_data, test_data = reader.read_all()
# vocab = reader.read_vocab()
# processor = QnliProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
# train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)
# test_dataset = processor.process_test(test_data)
#
# model = BaseSentencePairClassificationModel(plm="bert-base-cased", vocab=vocab)
# metric = BaseClassificationMetric(mode="binary")
# loss = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
#
# # evaluator = Evaluator(
# #     model=model,
# #     checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/simple_test1--2022-05-30--13-02-12.95/model/checkpoint-400",
# #     dev_data=dev_dataset,
# #     metrics=metric,
# #     sampler=None,
# #     drop_last=False,
# #     collate_fn=None,
# #     file_name="models.pt",
# #     batch_size=32,
# #     device=device,
# #     user_tqdm=True,
# # )
# # evaluator.evaluate()
# # print("End")
#
# trainer = Trainer(model,
#                   train_dataset,
#                   dev_data=dev_dataset,
#                   n_epochs=20,
#                   batch_size=32,
#                   loss=loss,
#                   optimizer=optimizer,
#                   scheduler=None,
#                   metrics=metric,
#                   train_sampler=None,
#                   dev_sampler=None,
#                   drop_last=False,
#                   gradient_accumulation_steps=1,
#                   num_workers=5,
#                   print_every=None,
#                   scheduler_steps=None,
#                   # checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/simple_test1--2022-05-30--13-02-12.95/model/checkpoint-300",
#                   validate_steps=None,      # validation setting
#                   save_steps=None,         # when to save model result
#                   output_path=output_path,
#                   grad_norm=1,
#                   use_tqdm=True,
#                   device=device,
#                   callbacks=None,
#                   metric_key=None,
#                   fp16=False,
#                   fp16_opt_level='O1',
#                   )
# trainer.train()
# print("end")
