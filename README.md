

<div align="center"><img src="./docs/source/figures/KTR-02-clip.png" width="350px" ></div>


<div align="center"r><b>A Knowledge Enhanced Text Representation Toolkit for Natural Language Understanding</b></div>


------------------



## Description

<div align=center><img width="450" height="250" src="./docs/source/figures/knowledge.png"/></div>


**CogKTR** is a **K**nowledge-enhanced **T**ext **R**epresentation toolkit for natural language understanding and has the following functions: 

+ CogKTR provides user-friendly knowledge acquisition interfaces. Users can use our toolkit to enhance the given texts with one click.

+ CogKTR supports various knowledge embeddings that can be used directly.  And we also implement plenty of knowledge-enhanced methods so researchers can quickly reproduce these models.

+ CogKTR supports many built-in NLU tasks to evaluate the effectiveness of knowledge-enhanced methods. In our paradigm, users can easily conduct their research via a pipeline.

+ We also release an [online demo](http://cognlp.com/cogktr/) to show the process of knowledge acquisition and the effect of knowledge enhancement. 

+ A short introduction video is available [here](https://youtu.be/SrvXrXdDiVY).


## Features

This easy-to-use python package has the following features:

+ **Unified** 
CogKTR is designed and built on our Unified Knowledge-Enhanced Paradigm, which consists of four stages: *knowledge acquisition*, *knowledge representation*, *knowledge injection*, and *knowledge application*.

+ **Knowledgeable** 
CogKTR integrates multiple knowledge sources, including [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page), [Wikipedia](https://www.wikipedia.org/), [ConceptNet](https://conceptnet.io/), [WordNet](https://wordnet.princeton.edu/)and [CogNet](http://cognet.top/index.html), and implements a series of knowledge-enhanced methods.

+ **Modular** 
CogKTR modularizes our proposed paradigm and consists of `Enhancer`, `Model`, `Core` and `Data` modules, each of which is highly extensible so that researchers can implement new components easily.

## Install

```shel
# clone CogKTR 
git clone https://github.com/CogNLP/CogKTR.git

# install CogKTR 
cd cogktr
pip install -e .   
pip install -r requirements.txt
```

## Quick Start 

### Text Matching with Base Model

```python
import torch.nn as nn
import torch.optim as optim
from cogktr import *

device, output_path = init_cogktr(
    device_id=6,
    output_path="path/to/store/the/experiments/results",
    folder_tag="base_classification",
)

reader = QnliReader(raw_data_path="path/storing/the/raw/dataset")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = QnliProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

plm = PlmBertModel(pretrained_model_name="bert-base-cased")
model = BaseSentencePairClassificationModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

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
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
print("end")

```

### Text Matching with Knowledge Enhanced Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *

from transformers import BertConfig
from argparse import Namespace


device, output_path = init_cogktr(
    device_id=9,
    output_path="path/to/store/the/experiments/results",
    folder_tag="base_classification_with_knowledge",
)

reader = QnliReader(raw_data_path="path/storing/the/raw/dataset")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

enhancer = LinguisticsEnhancer(load_ner=False,
                               load_srl=True,
                               load_syntax=False,
                               load_wordnet=False,
                               cache_path="path/to/cache/the/enhanced_data",
                               cache_file="cached_file_name",
                               reprocess=True)
                    
enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key_1="sentence",enhanced_key_2="question")
enhanced_dev_dict = enhancer.enhance_dev(dev_data,enhanced_key_1="sentence",enhanced_key_2="question")
enhanced_test_dict = enhancer.enhance_test(test_data,enhanced_key_1="sentence",enhanced_key_2="question")


processor = QnliSembertProcessor(plm="bert-base-uncased", max_token_len=256, vocab=vocab,debug=False)
train_dataset = processor.process_train(train_data,enhanced_train_dict)
dev_dataset = processor.process_dev(dev_data,enhanced_dev_dict)
test_dataset = processor.process_test(test_data,enhanced_test_dict)
early_stopping = EarlyStopping(mode="max",patience=3,threshold=0.001,threshold_mode="abs",metric_name="F1")

tag_config = {
   "tag_vocab_size":len(vocab["tag_vocab"]),
   "hidden_size":10,
   "output_dim":10,
   "dropout_prob":0.1,
   "num_aspect":3
}
tag_config = Namespace(**tag_config)
plm = SembertEncoder.from_pretrained("bert-base-uncased",tag_config=None)
model = SembertForSequenceClassification(
    vocab=vocab,
    plm=plm,
)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)


trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=16,
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
                  validate_steps=2000,  # validation setting
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
```

