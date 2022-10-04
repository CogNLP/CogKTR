
# Data for DPR searcher
The reference page can be found [here](
https://github.com/facebookresearch/DPR).

There are three kinds of data used by the dpr searcher.

+ **model_file** contains the checkpoint of the question encoder and wiki passage
    encoder and can be downloaded from [here](https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/single-adv-hn/nq/hf_bert_base.cp).
    
+ **index_path** contains the index of the wikipedia dense representations and can be constructed
using the command provided below after downloading all the required data and checkpoints mentioned
in section [retriever-validation-against-the-entire-set-of-documents](https://github.com/facebookresearch/DPR/blob/main/README.md#retriever-validation-against-the-entire-set-of-documents).

```shell
python dense_retriever.py \
model_file=/data/hongbang/projects/DPR/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp \
qa_dataset=nq_test \
ctx_datatsets=[dpr_wiki] \
encoded_ctx_files=["/data/hongbang/projects/DPR/downloads/data/retriever_results/nq/single-adv-hn/wikipedia_passages_*"] \
out_file=/data/hongbang/projects/DPR/outputs/nq_test_output.json \
index_path=/data/hongbang/projects/DPR/outputs/my_index/
```

+ **wiki_passages** contains the id, text and title of every wikipedia passages 
and can be downloaded from [here](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz).

All the data and checkpoints can be arranged in the following form
```angular2html
datapath
├─ knowledge_graph
│  ├─ dpr
│  │  ├─ bert-base-encoder.cp
│  │  ├── index_path
│  │  │   ├─ index.dpr
│  │  │   ├─ index_meta.dpr
│  │  └─ psgs_w100.tsv
│  │
```



