from .base_processor import *
from cogktr.data.processor.conll2003_processors import *
from cogktr.data.processor.conll2005_srl_subset_processors import *
from cogktr.data.processor.qnli_processors import *
from cogktr.data.processor.squad2_subset_processors import *
from cogktr.data.processor.sst2_processors import *
from cogktr.data.processor.stsb_processors import *
from cogktr.data.processor.squad2_processors import *

__all__ = [
    # baseprocessor
    "BaseProcessor",

    # conll2003processor
    "Conll2003Processor",

    # conll2005srisubsetprocessor
    "Conll2005SrlSubsetProcessor",

    # qnliprocessor
    "QnliProcessor",
    "QnliSembertProcessor",

    # squad2processor
    "Squad2Processor",
    "Squad2SembertProcessor",

    # squad2subsetprocessor
    "Squad2SubsetProcessor",

    # sst2processor
    "Sst2Processor",
    "Sst2ForKgembProcessor",
    "Sst2ForKtembProcessor",
    "Sst2ForSyntaxAttentionProcessor",
    "Sst2SembertProcessor",

    # stsbprocessor
    "StsbProcessor",

]
