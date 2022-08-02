from .base_processor import *
from cogktr.data.processor.commonsenseqa_processors import *
from cogktr.data.processor.commonsenseqa_qagnn_processors import *
from cogktr.data.processor.conll2003_processors import *
from cogktr.data.processor.lama_processors import *
from cogktr.data.processor.multisegchnsentibert_processors import *
from cogktr.data.processor.qnli_processors import *
from cogktr.data.processor.semcor_processors import *
from cogktr.data.processor.squad2_processors import *
from cogktr.data.processor.sst2_processors import *
from cogktr.data.processor.sst5_processors import *
from cogktr.data.processor.stsb_processors import *

__all__ = [
    # baseprocessor
    "BaseProcessor",

    # commonsenseqaprocessor
    "CommonsenseqaProcessor",

    # commonsenseqaqagnnprocessor
    "CommonsenseqaQagnnProcessor",

    # conll2003processor
    "Conll2003Processor",

    # lamaprocessor
    "LamaProcessor",

    # multisegchnsentibertprocessor
    "MultisegchnsentibertProcessor",
    "MultisegchnsentibertForHLGPresegProcessor",

    # qnliprocessor
    "QnliProcessor",
    "QnliSembertProcessor",

    # semcorprocessor
    "SemcorProcessor",
    "TSemcorProcessor",

    # squad2processor
    "Squad2Processor",
    "Squad2SembertProcessor",

    # sst2processor
    "Sst2Processor",
    "Sst2ForKgembProcessor",
    "Sst2ForKtembProcessor",
    "Sst2ForSyntaxAttentionProcessor",
    "Sst2SembertProcessor",

    # sst5processor
    "Sst5Processor",
    "Sst5ForKtattProcessor",

    # stsbprocessor
    "StsbProcessor",

]
