# The Natural Questions Dataset

Download The NaturalQustions dataset from here([train](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv)
,[dev](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv) and [test](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv)).

Put all the files path `datapath/question_answering/NaturalQuestions/raw_data`as
the following form:

 
```angular2html
datapath
├─ question_answering
│  ├─ NaturalQuestions
│  │  ├─ raw_data
│  │  │  │  ├─ nq-train.csv
│  │  │  │  ├─ nq-dev.csv
│  │  │  │  ├─ nq-test.csv
```

And use the following code to convert the original reading comprehension
model's state dict storage file so that it can be reused in our code.


```python
# Convert the original model weight file to adapt to current model
import torch
from collections import OrderedDict
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel

model_file = '/data/hongbang/projects/DPR/downloads/checkpoint/reader/nq-single/hf-bert-base.cp'
state_dict = torch.load(model_file)
model_state_dict = OrderedDict()
for key,value in state_dict["model_dict"].items():
    if not key.startswith("qa_classifier"):
        key_items = key.split(".")
        if key_items[0] == 'encoder':
            key_items[0] = 'bert'
        elif key_items[0] == 'qa_outputs':
            key_items[0] = 'linear'
        model_state_dict.update({".".join(key_items): value})

model = BaseReadingComprehensionModel(plm='bert-base-uncased',vocab=None)
model.to(torch.device('cuda:0'))
model.load_state_dict(model_state_dict,strict=False)
model.to(torch.device('cpu'))
torch.save(model.state_dict(),'/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data/bert-base-mrc-openqa.pt')
print("Done!")
```

All the files required for the evaluation of dataset Natural Questions should be
arranged like the following form:
```angular2html
datapath
├─ question_answering
│  ├─ NaturalQuestions
│  │  ├─ raw_data
│  │  │  │  ├─ nq-train.csv
│  │  │  │  ├─ nq-dev.csv
│  │  │  │  ├─ nq-test.csv
│  │  │  │  ├─ bert-base-mrc-openqa.pt
```


Reference project:https://github.com/facebookresearch/DPR.