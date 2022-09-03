from cogktr import *
import torch.nn as nn
import torch.optim as optim
from cogktr.models.sembert_model import SembertForSequenceClassification
from cogktr.modules.encoder.sembert import SembertEncoder

device, output_path = init_cogktr(
    device_id=8,
    output_path="/data/hongbang/CogKTR/datapath/text_classification/SST_2/experimental_result",
    folder_tag="sst2_sembert",
)


reader = Sst2Reader(raw_data_path="/data/hongbang/CogKTR/datapath/text_classification/SST_2/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = LinguisticsEnhancer(load_ner=False,
                               load_srl=True,
                               load_syntax=False,
                               load_wordnet=False,
                               cache_path="/data/hongbang/CogKTR/datapath/text_classification/SST_2/enhanced_data",
                               cache_file="linguistics_data",
                               reprocess=False)

enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key_1="sentence",return_srl=True)
enhanced_dev_dict = enhancer.enhance_dev(dev_data,enhanced_key_1="sentence",return_srl=True)
enhanced_test_dict = enhancer.enhance_test(test_data,enhanced_key_1="sentence",return_srl=True)

processor = Sst2SembertProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab,debug=False)
train_dataset = processor.process_train(train_data,enhanced_train_dict)
dev_dataset = processor.process_dev(dev_data,enhanced_dev_dict)

tag_config = {
   "tag_vocab_size":len(vocab["tag_vocab"]),
   "hidden_size":10,
   "output_dim":10,
   "dropout_prob":0.1,
   "num_aspect":3
}
plm = SembertEncoder.from_pretrained("bert-base-cased",tag_config=tag_config)
model = SembertForSequenceClassification(
    vocab=vocab,
    plm=plm,
)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
early_stopping = EarlyStopping(mode="max",patience=3,threshold=0.0001,threshold_mode="abs",metric_name="F1")

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=32,
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
                  validate_steps=500,
                  save_steps=None,
                  early_stopping=early_stopping,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
print("end")

