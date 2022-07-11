import torch.nn as nn
import torch.optim as optim

from cogktr import BaseClassificationMetric
from cogktr.data.processor.lama_processors import LamaProcessor
from cogktr.data.reader.lama_reader import LamaReader
from cogktr.models.base_question_answering_fill_mask_model import BaseQuestionAnsweringFillMaskModel
from cogktr.modules.plms.plm_maskedlm_bert import PlmMaskedLMBertModel
from cogktr.utils.general_utils import init_cogktr
from cogktr.core.trainer import Trainer

device, output_path = init_cogktr(
    device_id=0,
    output_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/question_answering/LAMA/experimental_result/",
    folder_tag="simple_test",
)

reader = LamaReader(
    raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/question_answering/LAMA/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = LamaProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

plm = PlmMaskedLMBertModel(pretrained_model_name="roberta-large")
model = BaseQuestionAnsweringFillMaskModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=10,
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
                  )
trainer.train()
print("end")
