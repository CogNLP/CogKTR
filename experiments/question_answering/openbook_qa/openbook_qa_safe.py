import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.io_utils import save_pickle,load_pickle
from cogktr.utils.general_utils import init_cogktr
from cogktr.data.processor.openbookqa_processors.openbookqa_for_safe_processor import OpenBookQAForSafeProcessor
from cogktr.models.safe_model import SAFEModel
from transformers import get_constant_schedule
from cogktr.enhancers.commonsense_enhancer import CommonsenseEnhancer

device, output_path = init_cogktr(
    device_id=7,
    output_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/",
    folder_tag="safe",
)

reader = OpenBookQAReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = CommonsenseEnhancer(
    knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet",
    cache_path='/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/',
    cache_file="enhanced_data",
    reprocess=False,
    load_conceptnet=True
)

enhanced_train_dict, enhanced_dev_dict, enhanced_test_dict = enhancer.enhance_all(train_data, dev_data, test_data,
                                                                                  vocab, return_metapath=True)

processor = OpenBookQAForSafeProcessor(
    plm="roberta-large",
    max_token_len=100,
    vocab=vocab,debug=False,
)
train_dataset = processor.process_train(train_data,enhanced_train_dict)
dev_dataset = processor.process_train(dev_data,enhanced_dev_dict)


plm = PlmAutoModel(pretrained_model_name="roberta-large")
model = SAFEModel(plm=plm)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=0.000001,weight_decay=0.001)
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
grouped_parameters = [
    {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.001, 'lr': 1e-5},
    {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0, 'lr': 1e-5},
    {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.001, 'lr': 0.01},
    {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0, 'lr': 0.01},
]
optimizer = optim.RAdam(grouped_parameters)
scheduler = get_constant_schedule(optimizer)
early_stopping = EarlyStopping(mode="max",patience=10,threshold=0.001,threshold_mode="abs",metric_name="Acc")

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=100,
                  batch_size=8,
                  gradient_accumulation_steps=16,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  save_by_metric="Acc",
                  early_stopping=early_stopping,
                  validate_steps=None,
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1.0,
                  use_tqdm=True,
                  device=device,
                  callbacks=None,
                  metric_key=None,
                  fp16=False,
                  fp16_opt_level='O1',
                  # collate_fn=train_dataset.to_dict,
                  # dev_collate_fn=dev_dataset.to_dict,
                  )
trainer.train()
print("end")

