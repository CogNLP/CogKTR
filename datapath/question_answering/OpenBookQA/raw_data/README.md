# The OpenBookQA Dataset

Download The OpenBookQA Dataset from https:https://github.com/RUCAIBox/SAFE/blob/main/script/download_raw_data.sh After downloading
the original zip file `OpenBookQA-V1-Sep2018.zip`,unzip it, extract all the files from directory `OpenBookQA-V1-Sep2018/Data/Main`
and put it in path`datapath/question_answering/OpenBookQA/raw_data` as the following form:

```angular2html
datapath
├─ question_answering
│  ├─ OpenBookQA
│  │  ├─ raw_data
│  │  │  │  ├─ train.jsonl
│  │  │  │  ├─ train.tsv
│  │  │  │  ├─ dev.jsonl
│  │  │  │  ├─ dev.tsv
│  │  │  │  ├─ test.jsonl
│  │  │  │  ├─ test.tsv
│  │  │  │  ├─ openbook.txt
 
```