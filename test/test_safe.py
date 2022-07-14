import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.io_utils import save_pickle,load_pickle
from cogktr.utils.general_utils import init_cogktr
from cogktr.data.processor.openbookqa_processors.openbookqa_for_safe_processor import OpenBookQAForSafeProcessor
from cogktr.enhancers.conceptnet_enhancer import ConceptNetEnhancer
from cogktr.models.safe_model import SAFEModel

device, output_path = init_cogktr(
    device_id=7,
    output_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/",
    folder_tag="safe_debug_1e-6_epoch100",
)

reader = OpenBookQAReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = ConceptNetEnhancer(
        knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph/conceptnet",
        cache_path='/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/enhanced_data',
        reprocess=False
    )

enhanced_train_dict,enhanced_dev_dict,enhanced_test_dict = enhancer.enhance_all(train_data,dev_data,test_data,vocab)

processor = OpenBookQAForSafeProcessor(
    plm="roberta-large",
    max_token_len=100,
    vocab=vocab
)
train_dataset = processor.process_train(train_data,enhanced_train_dict)
dev_dataset = processor.process_train(dev_data,enhanced_dev_dict)


plm = PlmAutoModel(pretrained_model_name="roberta-large")
model = SAFEModel(plm=plm,vocab=None)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001,weight_decay=0.001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=100,
                  batch_size=3,
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
                  # collate_fn=train_dataset.to_dict,
                  # dev_collate_fn=dev_dataset.to_dict,
                  )
trainer.train()
print("end")

