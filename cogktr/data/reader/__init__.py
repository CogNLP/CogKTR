from .base_reader import *
from .commonsenseqa_reader import *
from .commonsenseqa_qagnn_reader import *
from .conll2003_reader import *
from .lama_reader import *
from .multisegchnsentibert_reader import *
from .qnli_reader import *
from .semcor_reader import *
from .squad2_reader import *
from .sst2_reader import *
from .sst5_reader import *
from .stsb_reader import *
from .openbookqa_reader import *

__all__ = [
    "BaseReader",
    "CommonsenseqaReader",
    "CommonsenseqaQagnnReader",
    "OpenBookQAReader",
    "Conll2003Reader",
    "LamaReader",
    "MultisegchnsentibertReader",
    "QnliReader",
    "SemcorReader",
    "Squad2Reader",
    "Sst2Reader",
    "Sst5Reader",
    "StsbReader",
]
