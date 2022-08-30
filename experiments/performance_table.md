# Performance Table

Note:For the first time to run the program, please set the Enhancer reprocess parameters as True.

## 1.Text Classification

### 1.1 SST-2 (DEV)

| Model | P↑ | R↑ | F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|
| bert-base-cased(paper) | \ | \ | \ | 0.922 | sst2_bert_base_cased.py |
| KT-Emb+bert-base-cased | 0.92222 | 0.93468 | 0.92841 | 0.92660 | sst2_ktemb_bert_base_cased.py |
| KG-Emb+bert-base-cased | 0.92792 | 0.92792 | 0.92792 | 0.92660 | sst2_kgemb_bert_base_cased.py |


### 1.2 SST-5 (DEV)

| Model | micro_P↑ | micro_R↑ | micro_F1↑ | macro_P↑ | macro_R↑ | macro_F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|---|---|---|
| bert-base-cased | 0.48319 | 0.48319 | 0.48319 | 0.48459 | 0.46527 | 0.47156 | 0.48319 | sst5_bert_base_cased.py |
| sembert | 0.49137 | 0.49137 | 0.49137 | 0.5108 | 0.47076 | 0.47388 | 0.49137 | sst5_sembert.py |

### 1.3 MultiSegChnSentiBERT (TEST)

| Model | P↑ | R↑ | F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|
| bert-base-chinese(paper) | \ | \ | 0.9472 | \ | multisegchnsentibert_bert_base_chinese.py |
| HLG+bert-base-chinese+jieba | 0.96795 | 0.94407 | 0.95587 | 0.95583 | multisegchnsentibert_hlg_bert_base_chinese.py |
| HLG+bert-base-chinese+pre-seg | 0.97762 | 0.93421 | 0.95542 | 0.95583 | multisegchnsentibert_hlg_pre_seg_bert_base_chinese.py |

## 2.Sentence Pair

### 2.1 QNLI (DEV)

| Model | P↑ | R↑ | F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|
| bert-base-cased | 0.90824 | 0.91416 | 0.91119 | 0.90993 | qnli_bert_base_cased.py |

### 2.2 STS-B (DEV)

| Model | r2↑(key) | mse↓(key) | mae | pear | code |
|---|---|---|---|---|---|
| bert-base-cased | 0.76615 | 0.51084 | 0.53663 | 0.88735 | stsb_bert_base_cased.py |

## 3.Sequence Labeling

### 3.1 conll2003 (DEV)

| Model | P↑ | R↑ | F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|
| bert-base-cased | 0.94923 | 0.95658 | 0.95289 | 0.91002 | conll2003_bert_base_cased.py |

## 4.Question Answering

### 4.1 CommonsenseQA (DEV)

| Model | micro_P↑ | micro_R↑ | micro_F1↑ | macro_P↑ | macro_R↑ | macro_F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|---|---|---|
| bert-base-cased | 0.55855 | 0.55855 | 0.55855 | 0.55858 | 0.55801 |  0.55816 | 0.55855 | commonsense_qa_bert_base_cased.py |
| roberta-large | 0.75184 | 0.75184 | 0.75184 | 0.75204 | 0.78954 | 0.75169 | 0.75184 | commonsense_qa_roberta_large.py |
| SAFE | 0.7633 | 0.7633 | 0.7633 | 0.7631 | 0.7628 | 0.7629 | 0.76331 | commonsense_qa_safe.py |

[comment]: <> "| QAGNN | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"


### 4.2 OpenbookQA &#40;DEV&#41;

| Model | micro_P↑ | micro_R↑ | micro_F1↑ | macro_P↑ | macro_R↑ | macro_F1↑(key) | Acc↑ | code |
|---|---|---|---|---|---|---|---|---|
| roberta-large | 0.644 | 0.644 | 0.644 | 0.6443   | 0.6442 | 0.6426 | 0.644 | openbook_qa_roberta_large.py |
| SAFE | 0.664 | 0.664 | 0.664 | 0.6647 | 0.6657 | 0.6636 | 0.664 | openbook_qa_safe.py |

[comment]: <> "| QAGNN | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"

 


## 5.Disambiguation

### 5.1 Semcor (semeval2007)

| Model | macro_F1↑(key) | code |
|---|---|---|
| bert-base-cased+defination | 73.40659 | semcor_bert_base_cased.py |
| bert-base-cased+defination+hypernyms | 74.72527 | semcor_esr_bert_base_cased.py |


[comment]: <> "## 6.Reading Comprehension"

[comment]: <> "### 6.1 NULL"


[comment]: <> "## 7.Masked LM"

[comment]: <> "### 7.1 NULL"