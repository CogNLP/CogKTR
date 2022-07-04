from .base_processor import *
from cogktr.data.processor.commonsenseqa_qagnn_processors import *
from cogktr.data.processor.conll2003_processors import *
from cogktr.data.processor.multisegchnsentibert_processors import *
from cogktr.data.processor.qnli_processors import *
from cogktr.data.processor.sst2_processors import *
from cogktr.data.processor.stsb_processors import *
from cogktr.data.processor.squad2_processors import *

__all__ = [
    # baseprocessor
    "BaseProcessor",

    #commonsenseqaqagnnprocessor
    "CommonsenseqaQagnnProcessor",

    # conll2003processor
    "Conll2003Processor",

    # multisegchnsentibertprocessor
    "MultisegchnsentibertProcessor",

    # qnliprocessor
    "QnliProcessor",
    "QnliSembertProcessor",

    # squad2processor
    "Squad2Processor",
    "Squad2SembertProcessor",

    # sst2processor
    "Sst2Processor",
    "Sst2ForKgembProcessor",
    "Sst2ForKtembProcessor",
    "Sst2ForSyntaxAttentionProcessor",
    "Sst2SembertProcessor",

    # stsbprocessor
    "StsbProcessor",

]
