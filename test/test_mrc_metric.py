import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.core.metric.base_reading_comprehension_metric import BaseMRCMetric
from cogktr.data.processor.squad2_processors.squad2_processor import Squad2Processor
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel
from cogktr.utils.io_utils import save_pickle,load_pickle

device,output_path = init_cogktr(
    device_id=9,
    output_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/experimental_result/",
    folder_tag="mrc_metric_evaluation",
)

reader = Squad2Reader(raw_data_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
train_data, dev_data, _ = reader.read_all()
vocab = reader.read_vocab()

processor = Squad2Processor(plm="bert-base-uncased", max_token_len=384, vocab=vocab,debug=False)
# train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
# dev_cache_file = "/data/hongbang/CogKTR/test/test_squad_output/dev_dataset.pkl"
# save_pickle(dev_dataset,dev_cache_file)
# dev_dataset = load_pickle(dev_cache_file)
# print("Finished Saving!")

model = BaseReadingComprehensionModel(plm="bert-base-uncased",vocab=vocab)
metric = BaseMRCMetric()

evaluator = Evaluator(
    model=model,
    # checkpoint_path="/data/hongbang/projects/Learning/models/",
    checkpoint_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/experimental_result/mrc_baseline_5e-06--2022-09-29--01-53-54.67/best_model/checkpoint-288000",
    dev_data=dev_dataset,
    metrics=metric,
    sampler=None,
    drop_last=False,
    collate_fn=processor._collate,
    file_name="models.pt",
    batch_size=32,
    device=device,
    user_tqdm=True,
)
evaluator.evaluate()
print("End")






