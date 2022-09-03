import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.core.metric.base_reading_comprehension_metric import BaseMRCMetric
from cogktr.data.processor.squad2_processors.squad2_processor import Squad2Processor
from cogktr.models.base_reading_comprehension_model import BaseReadingComprehensionModel


enhancer = LinguisticsEnhancer(load_ner=False,
                               load_srl=False,
                               load_syntax=True,
                               load_wordnet=False,
                               cache_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/enhanced_data",
                               cache_file="syntax_data",
                               reprocess=True)

reader = Squad2Reader(raw_data_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
train_data, dev_data, _ = reader.read_all()
vocab = reader.read_vocab()

# debug process
question_text = dev_data["question_text"][0]
context_text = dev_data["context_text"][0]
from cogktr.enhancers.tagger.syntax_tagger import SyntaxTagger
tagger = SyntaxTagger(tool="stanza")
tagger_dict_1 = tagger.tag(question_text)
tagger_dict_2 = tagger.tag(context_text)

def convert_head_to_span(all_heads):
    hpsg_lists = []
    for heads in all_heads:
        n = len(heads)
        childs = [[] for i in range(n + 1)]
        left_p = [i for i in range(n + 1)]
        right_p = [i for i in range(n + 1)]

        def dfs(x):
            for child in childs[x]:
                dfs(child)
                left_p[x] = min(left_p[x], left_p[child])
                right_p[x] = max(right_p[x], right_p[child])

        for i, head in enumerate(heads):
            childs[head].append(i + 1)

        dfs(0)
        hpsg_list = []
        for i in range(1, n + 1):
            hpsg_list.append((left_p[i], right_p[i]))

        hpsg_lists.append(hpsg_list)

    return hpsg_lists
question_text_heads = [tagger_dict_1["knowledge"]["heads"]]
context_text_heads = [tagger_dict_2["knowledge"]["heads"]]

question_text_spans = convert_head_to_span(question_text_heads)
context_text_spans = convert_head_to_span(context_text_heads)


# enhanced_train_dict = enhancer.enhance_train(train_data,enhanced_key_1="question_text",enhanced_key_2="doc_tokens",return_syntax=True)
# enhanced_dev_dict = enhancer.enhance_dev(dev_data,enhanced_key_1="question_text",enhanced_key_2="doc_tokens",return_syntax=True)

# processor = Squad2Processor(plm="bert-base-cased", max_token_len=512, vocab=vocab,debug=False)
# train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)



