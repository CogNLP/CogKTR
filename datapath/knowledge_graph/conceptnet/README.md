# Conceptnet Data
Download the data from https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
 and unzip it using command `gzip -d conceptnet-assertions-5.6.0.csv.gz`

Download the pretrained-embedding file from https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy

Put the raw data in path datapath/knowledge_graph/conceptnet as
the following form:
```angular2html
datapath
├─ knowledge_graph
│  ├─ conceptnet
│  │  ├─ conceptnet-assertions-5.6.0.csv
│  │  ├─ tzw.ent.npy
```