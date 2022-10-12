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

Reference project:https://github.com/facebookresearch/DPR.