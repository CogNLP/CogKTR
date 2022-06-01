TABLE_DATA_TAGGER = [
    [
        "Tagger",
        "PosTagger",
        "Flair",
        "tokens,labels",
    ],
    [
        "Tagger",
        "NerTagger",
        "Flair",
        "tokens,labels",
    ],
    [
        "Tagger",
        "NerTagger",
        "CogIE",
        "None",
    ],
    [
        "Tagger",
        "SpoTagger",
        "CogIE",
        "None",
    ],
    [
        "Tagger",
        "SrlTagger",
        "Allennlp",
        "tokens,arguments,roles",
    ],
    [
        "Tagger",
        "SrlTagger",
        "CogIE",
        "None",
    ],
    [
        "Tagger",
        "EventTagger",
        "CogIE",
        "None",
    ],
]


TABLE_DATA_LINKER=[
    [
        "Linker",
        "WikipediaLinker",
        "Tagme",
        "begins,ends,ids,titles,mentions",
    ],
    [
        "Linker",
        "WikipediaLinker",
        "CogIE",
        "None",
    ],
]

TABLE_DATA_SEARCHER=[
    [
        "Searcher",
        "WikipediaSearcher",
        "Local Blink",
        "desc",
    ],
    [
        "Searcher",
        "WikipediaSearcher",
        "KILT",
        "None",
    ],
]

TABLE_DATA_EMBEDDER=[
    [
        "Embedder",
        "WikipediaEmbedder",
        "Wikipedia2vec",
        "en_embed,similar_embed",
    ],
    [
        "Embedder",
        "WikipediaEmbedder",
        "CogKGE",
        "None",
    ],
]