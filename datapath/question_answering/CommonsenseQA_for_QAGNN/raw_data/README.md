# The CommonsenseQA for QA-GNN Dataset

Download The CommonsenseQA_for_QAGNN Dataset from https://github.com/michiyasunaga/qagnn/blob/main/download_preprocessed_data.sh.    
Put the raw data in path`datapath/question_answering/CommonsenseQA_for_QAGNN/raw_data`.   
Like the following form:

```angular2html
datapath
├─ question_answering
│  ├─ CommonsenseQA_for_QAGNN
│  │  ├─ raw_data
│  │  │  ├─ cpnet
│  │  │  │  ├─ tzw.ent.npy
│  │  │  │  ├─ matcher_patterns.json
│  │  │  │  ├─ conceptnet.en.unpruned.graph
│  │  │  │  ├─ conceptnet.en.pruned.graph
│  │  │  │  ├─ conceptnet.en.csv
│  │  │  │  ├─ conceptnet-assertions-5.6.0.csv
│  │  │  │  └─ concept.txt
│  │  │  ├─ graph
│  │  │  │  ├─ train.graph.adj.pk.loaded_cache
│  │  │  │  ├─ train.graph.adj.pk
│  │  │  │  ├─ test.graph.adj.pk.loaded_cache
│  │  │  │  ├─ test.graph.adj.pk
│  │  │  │  ├─ dev.graph.adj.pk.loaded_cache
│  │  │  │  └─ dev.graph.adj.pk
│  │  │  ├─ grounded
│  │  │  │  ├─ train.grounded.jsonl
│  │  │  │  ├─ test.grounded.jsonl
│  │  │  │  └─ dev.grounded.jsonl
│  │  │  ├─ statement
│  │  │  │  ├─ train.statement.jsonl
│  │  │  │  ├─ test.statement.jsonl
│  │  │  │  └─ dev.statement.jsonl
│  │  │  ├─ train_rand_split.jsonl
│  │  │  ├─ dev_rand_split.jsonl
│  │  │  ├─ test_rand_split_no_answers.jsonl
│  │  │  └─ inhouse_split_qids.txt
│  │  │
```