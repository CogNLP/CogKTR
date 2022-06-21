from .base_tagger import *
from .ner_tagger import *
from .pos_tagger import *
from .srl_tagger import *
from .syntax_tagger import *

__all__ = [
    "BaseTagger",
    "NerTagger",
    "PosTagger",
    "SrlTagger",
    "SyntaxTagger",
    #TODO:add SpoTagger EventTagger
]
