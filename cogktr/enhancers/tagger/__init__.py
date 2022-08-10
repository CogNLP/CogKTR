from .base_tagger import *
from .ner_tagger import *
from .srl_tagger import *
from .syntax_tagger import *
from .word_segmentation_tagger import *
__all__ = [
    "BaseTagger",
    "NerTagger",
    "SrlTagger",
    "SyntaxTagger",
    "WordSegmentationTagger",
    #TODO:add SpoTagger EventTagger
]
