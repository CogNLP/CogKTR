import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.data.processor.sst2_processors.sst2_for_ebert_processor import Sst2ForEbertProcessor
from cogktr.enhancers.new_enhancer import NewEnhancer
from cogktr.utils.general_utils import init_cogktr
from cogktr.utils.constant.kbert_constants.constants import *
from cogktr.data.processor.sst2_processors.sst2_for_kbert_processor import *
from cogktr.models.kbert_model import KBertForSequenceClassification

# initiate
device, output_path = init_cogktr(
    device_id=0,
    output_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/experimental_result",
    folder_tag="simple_test",
)
# device = torch.device("cpu")

# Load the data
reader = Sst2Reader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

# enhance
enhancer = NewEnhancer(config_path="/home/chenyuheng/zhouyuyang/CogKTR/cogktr/utils/config/enhancer_config.json",
                       knowledge_graph_path="/home/chenyuheng/zhouyuyang/CogKTR/cogktr/datapath/knowledge_graph",
                       save_file_name="pre_enhanced_data",
                       enhanced_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/enhanced_data",
                       load_wikipedia=True,
                       reprocess=True)
enhanced_train_dict = enhancer.enhance_train(train_data, return_wikipedia=True)
enhanced_dev_dict = enhancer.enhance_dev(dev_data, return_wikipedia=True)
enhanced_test_dict = enhancer.enhance_test(test_data, return_wikipedia=True)

# processor
processor = Sst2ForEbertProcessor(plm="bert-base-uncased",
                                  device=device,
                                  matrix_load_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikipedia_emb_2_bert_emb/mapping.pt")
train_dataset = processor.process_train(data=train_data, enhanced_dict=enhanced_train_dict)
dev_dataset = processor.process_dev(data=dev_data, enhanced_dict=enhanced_dev_dict)
test_dataset = processor.process_test(data=test_data, enhanced_dict=enhanced_test_dict)

# model
model = PlmBertModel("bert-base-uncased")
# TODO: seperate plm and classification model
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# train
trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=50,
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
                  validate_steps=100,
                  save_steps=100,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  callbacks=None,
                  metric_key=None,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=train_dataset.to_dict,
                  dev_collate_fn=dev_dataset.to_dict,
                  )
trainer.train()
print("end")
