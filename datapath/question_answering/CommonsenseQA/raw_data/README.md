# The CommonsenseQA Dataset

Download The CommonsenseQA_for_QAGNN Dataset from https://github.com/michiyasunaga/qagnn/blob/main/download_raw_data.sh.    
Put the raw data in path`datapath/question_answering/CommonsenseQA/raw_data`.   
Like the following form:

```angular2html
datapath
├─ question_answering
│  ├─ CommonsenseQA_for_QAGNN
│  │  ├─ raw_data
│  │  │  ├─ train.statement.jsonl
│  │  │  ├─ test.statement.jsonl
│  │  │  ├─ dev.statement.jsonl
│  │  │  ├─ train_rand_split.jsonl
│  │  │  ├─ dev_rand_split.jsonl
│  │  │  ├─ test_rand_split_no_answers.jsonl
│  │  │  └─ inhouse_split_qids.txt
│  │  │
```