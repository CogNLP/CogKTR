# CUDA_VISIBLE_DEVICES="2,3,4,5"  python -m torch.distributed.launch --nproc_per_node 4 test_base_sentence_pair_classification.py
# CUDA_VISIBLE_DEVICES="5,6,7,9"  python -m torch.distributed.launch --nproc_per_node 4 ddp_base_sembert.py

import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn as nn
from argparse import Namespace

from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.utils.general_utils import init_cogktr
from cogktr.models.sembert_model import SembertForSequenceClassification
from cogktr.modules.encoder.sembert import SembertEncoder

def main(local_rank):
    device,output_path = init_cogktr(
        output_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/",
        folder_tag="sembert_wit_tag_multi_gpu",
        rank=local_rank,
        seed=1 + local_rank,
    )
    if local_rank != 0:
        dist.barrier()
    reader = QnliReader(raw_data_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    if local_rank == 0:
        dist.barrier()

    enhancer = Enhancer(reprocess=False,
                        return_srl=True,
                        save_file_name="pre_enhanced_data",
                        datapath="/data/hongbang/CogKTR/datapath",
                        enhanced_data_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/enhanced_data")
    enhanced_train_dict = enhancer.enhance_train(train_data, enhanced_key_1="sentence", enhanced_key_2="question")
    enhanced_dev_dict = enhancer.enhance_dev(dev_data, enhanced_key_1="sentence", enhanced_key_2="question")
    enhanced_test_dict = enhancer.enhance_test(test_data, enhanced_key_1="sentence", enhanced_key_2="question")

    processor = QnliSembertProcessor(plm="bert-base-uncased", max_token_len=256, vocab=vocab, debug=False)
    train_dataset = processor.process_train(train_data, enhanced_train_dict)
    dev_dataset = processor.process_dev(dev_data, enhanced_dev_dict)

    train_sampler = DistributedSampler(train_dataset)
    dev_sampler = DistributedSampler(dev_dataset)

    tag_config = {
        "tag_vocab_size": len(vocab["tag_vocab"]),
        "hidden_size": 10,
        "output_dim": 10,
        "dropout_prob": 0.1,
        "num_aspect": 3
    }
    tag_config = Namespace(**tag_config)
    plm = SembertEncoder.from_pretrained("bert-base-uncased", tag_config=tag_config)
    model = SembertForSequenceClassification(
        vocab=vocab,
        plm=plm,
    )
    metric = BaseClassificationMetric(mode="binary")
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    early_stopping = EarlyStopping(mode="max", patience=3, threshold=0.001, threshold_mode="abs", metric_name="F1")

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
                      validate_steps=2000,      # validation setting
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