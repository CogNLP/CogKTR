# The OpenBookQA Dataset

Download The OpenBookQA Dataset from https:https://github.com/RUCAIBox/SAFE/blob/main/script/download_raw_data.sh
Put the raw data in path`datapath/question_answering/OpenBookQA/raw_data`.   
Like the following form:

```angular2html
datapath
├─ question_answering
│  ├─ OpenBookQA
│  │  ├─ raw_data
│  │  │  ├─ OpenBookQA-V1-Sep2018
│  │  │  │  ├─ Data
│  │  │  │  │  ├─ Additional
│  │  │  │  │  │  ├─ crowdsourced-facts.txt
│  │  │  │  │  │  ├─ train_complete.jsonl
│  │  │  │  │  │  ├─ test_complete.jsonl
│  │  │  │  │  │  ├─ dev_complete.jsonl
│  │  │  │  │  ├─ Main
│  │  │  │  │  │  ├─ train.jsonl
│  │  │  │  │  │  ├─ train.tsv
│  │  │  │  │  │  ├─ dev.jsonl
│  │  │  │  │  │  ├─ dev.tsv
│  │  │  │  │  │  ├─ test.jsonl
│  │  │  │  │  │  ├─ test.tsv
│  │  │  │  │  │  ├─ openbook.txt
 
```