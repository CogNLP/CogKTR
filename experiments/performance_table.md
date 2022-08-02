
#  Performance Table
Note:For the first time to run the program, please set the Enhancer reprocess parameters as True.

## 1.Text Classification

### 1.1 SST-2 (DEV)

| Model | P↑ | R↑ | F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|
| bert-base-cased | 0.90343 | 0.94819 | 0.92527 | 0.92201 | sst2_bert_base_cased.py |
| KT-Emb+bert-base-cased | 0.92222 | 0.93468 | 0.92841 | 0.92660 | sst2_ktemb_bert_base_cased.py |
| KG-Emb+bert-base-cased | 0.92792 | 0.92792 | 0.92792 | 0.92660 | sst2_kgemb_bert_base_cased.py |

### 1.2 SST-5 (DEV)

| Model | micro_P↑ | micro_R↑ | micro_F1↑ | macro_P↑ | macro_R↑ | macro_F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|---|---|---|
| bert-base-cased | 0.48319 | 0.48319 | 0.48319 | 0.48459 | 0.46527 | 0.47156 | 0.48319 | sst5_bert_base_cased.py | 
| KT-Emb+bert-base-cased | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 | 
| KG-Emb+bert-base-cased | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 |

### 1.3 MultiSegChnSentiBERT

| Model | P↑ | R↑ | F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|
| bert-base-chinese | 0 | 0 | 0 | 0 | 0 |
| HLG | 8 | 9 | 6 | 6 | 0 |

## 2.Sentence Pair

### 2.1 QNLI (DEV)

| Model | P↑ | R↑ | F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|
| bert-base-cased | 0.90824 | 0.91416 | 0.91119 | 0.90993 | qnli_bert_base_cased.py |
| KT-Emb+bert-base-cased | 8 | 9 | 6 | 6 | 6 |
| KG-Emb+bert-base-cased | 8 | 9 | 6 | 6 | 6 |

### 2.2 STS-B (DEV)

| Model | r2↑(key) | mse↓(key) | mae | pear | code |
|---|---|---|---|---|---|
| bert-base-cased | 0.76615 | 0.51084 | 0.53663 | 0.88735 | stsb_bert_base_cased.py | 
| KT-Emb+bert-base-cased | 8 | 6 | 5 | 6 | 6 | 
| KG-Emb+bert-base-cased | 8 | 6 | 5 | 6 | 6 |

## 3.Sequence Labeling

### 3.1 conll2003 (DEV)

| Model | P↑ | R↑ | F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|
| bert-base-cased | 0.94923 | 0.95658 | 0.95289 | 0.91002 | conll2003_bert_base_cased.py |
| KT-Emb+bert-base-cased | 8 | 9 | 6 | 6 | 6 |
| KG-Emb+bert-base-cased | 8 | 9 | 6 | 6 | 6 |

## 4.Question Answering

### 4.1 CommonsenseQA (DEV)

| Model | micro_P↑ | micro_R↑ | micro_F1↑ | macro_P↑ | macro_R↑ | macro_F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|---|---|---|
| bert-base-cased | 0.55855 | 0.55855 | 0.55855 | 0.55858 | 0.55801 |  0.55816 | 0.55855 | commonsense_qa_bert_base_cased.py | 
| QAGNN | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 | 
| SAFE | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 |

### 4.2 OpenbookQA (DEV)

| Model | micro_P↑ | micro_R↑ | micro_F1↑ | macro_P↑ | macro_R↑ | macro_F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|---|---|---|
| bert-base-cased | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| QAGNN | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 | 
| SAFE | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 |


## 5.Disambiguation

### 5.1 Semcor (semeval2007)

| Model | micro_P↑ | micro_R↑ | micro_F1↑ | macro_P↑ | macro_R↑ | macro_F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|---|---|---|
| bert-base-cased | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| ESR | 8 | 9 | 4 | 5 | 6 | 5 | 6 | 6 |


## 6.Reading Comprehension

### 6.1 NULL


## 7.Masked LM

### 7.1 NULL