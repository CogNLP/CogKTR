#Performance Table

## 1.Text Classification

### 1.1 SST-2 (DEV)

| Model | P | R | F1 | Acc | file |
|---|---|---|---|---|---|
| bert-base-cased | 0.90343 | 0.94819 | 0.92527 | 0.92201 | sst2_bert_base_cased.py |
| bert-base-cased+KT-Emb | 8 | 9 | 6 | 6 | 6 |
| bert-base-cased+KG-Emb | 8 | 9 | 6 | 6 | 6 |
| bert-base-cased+KT-Attn | 8 | 9 | 6 | 6 | 6 |

### 1.2 SST-5 (DEV)

| Model | micro_P | micro_R | micro_F1 | macro_P | macro_R | macro_F1 | Acc | file |
|---|---|---|---|---|---|---|---|---|
| bert-base-cased | 0.48319 | 0.48319 | 0.48319 | 0.48459 | 0.46527 | 0.47156 | 0.48319 | sst5_bert_base_cased.py | 
| bert-base-cased+KT-Emb | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 | 
| bert-base-cased+KG-Emb | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 | 
| bert-base-cased+KT-Attn | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 | 

## 2.Sentence Pair