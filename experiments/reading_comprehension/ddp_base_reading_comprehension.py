# CUDA_VISIBLE_DEVICES="2,3,4,5"  python -m torch.distributed.launch --nproc_per_node 4 test_base_sentence_pair_classification.py

import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn as nn

from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.utils.general_utils import init_cogktr
from cogktr.core.metric.base_reading_comprehension_metric import BaseMRCMetric
from cogktr.data.processor.squad2_processors.squad2_processor import Squad2Processor
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel

def main(local_rank):
    device,output_path = init_cogktr(
        output_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/experimental_result/",
        folder_tag="mrc_baseline_multi_gpu",
        rank=local_rank,
        seed=1 + local_rank,
    )
    if local_rank != 0:
        dist.barrier()
    reader = Squad2Reader(raw_data_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
    train_data, dev_data, _ = reader.read_all()
    vocab = reader.read_vocab()
    if local_rank == 0:
        dist.barrier()

    processor = Squad2Processor(plm="bert-base-cased", max_token_len=512, vocab=vocab, debug=False)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)

    train_sampler = DistributedSampler(train_dataset)
    dev_sampler = DistributedSampler(dev_dataset)

    model = BaseReadingComprehensionModel(plm="bert-base-cased", vocab=vocab)
    metric = BaseMRCMetric()
    loss = nn.CrossEntropyLoss(ignore_index=512)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    early_stopping = EarlyStopping(mode="max", patience=8, threshold=0.01, threshold_mode="abs", metric_name="F1")

    dist.barrier()

    trainer = Trainer(model,
                      train_dataset,
                      dev_data=dev_dataset,
                      n_epochs=100,
                      batch_size=16,
                      loss=loss,
                      optimizer=optimizer,
                      scheduler=None,
                      metrics=metric,
                      train_sampler=train_sampler,
                      dev_sampler=dev_sampler,
                      early_stopping=early_stopping,
                      save_by_metric="F1",
                      drop_last=False,
                      gradient_accumulation_steps=1,
                      num_workers=5,
                      print_every=None,
                      scheduler_steps=None,
                      validate_steps=1000,      # validation setting
                      save_steps=None,         # when to save model result
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