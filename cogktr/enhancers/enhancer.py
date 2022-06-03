from rich.console import Console
from rich.table import Table
from cogktr.enhancers.tagger import PosTagger, NerTagger, SrlTagger
from cogktr.enhancers.linker import WikipediaLinker
from cogktr.enhancers.searcher import WikipediaSearcher
from cogktr.enhancers.embedder import WikipediaEmbedder
from cogktr.utils.constant_utils import TABLE_DATA_TAGGER, TABLE_DATA_LINKER, TABLE_DATA_SEARCHER, TABLE_DATA_EMBEDDER


class Enhancer:
    def __init__(self,
                 return_pos=False,
                 return_ner=False,
                 return_spo=False,
                 return_srl=False,
                 return_event=False,
                 return_entity_desc=False,
                 return_entity_ebd=False):
        self.return_pos = return_pos
        self.return_ner = return_ner
        self.return_spo = return_spo
        self.return_srl = return_srl
        self.return_event = return_event
        self.return_entity_desc = return_entity_desc
        self.return_entity_ebd = return_entity_ebd

        self._init_config()
        self._init_module()

    def _init_config(self):
        self.tool_config = {}
        self.tool_config["PosTagger"] = "flair"
        self.tool_config["NerTagger"] = "flair"
        self.tool_config["SpoTagger"] = None
        self.tool_config["SrlTagger"] = "allennlp"
        self.tool_config["EventTagger"] = None
        self.tool_config["WikipediaLinker"] = "tagme"
        self.tool_config["WikipediaSearcher"] = "blink"
        self.tool_config["WikipediaEmbedder"] = "wikipedia2vec"

        self.path_config = {}
        self.path_config[
            "WikipediaSearcher"] = "/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia/raw_data/entity.jsonl"
        self.path_config[
            "WikipediaEmbedder"] = "/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl"

    def _init_module(self):
        if self.return_pos:
            self.pos_tagger = PosTagger(tool=self.tool_config["NerTagger"])
        if self.return_ner:
            self.ner_tagger = NerTagger(tool=self.tool_config["NerTagger"])
        if self.return_srl:
            self.srl_tagger = SrlTagger(tool=self.tool_config["SrlTagger"])
        if self.return_entity_desc or self.return_entity_ebd:
            self.wikipedia_linker = WikipediaLinker(tool=self.tool_config["WikipediaLinker"])
        if self.return_entity_desc:
            self.wikipedia_searcher = WikipediaSearcher(tool=self.tool_config["WikipediaSearcher"],
                                                        path=self.path_config["WikipediaSearcher"])
        if self.return_entity_ebd:
            self.wikipedia_embedder = WikipediaEmbedder(tool=self.tool_config["WikipediaEmbedder"],
                                                        path=self.path_config["WikipediaEmbedder"])

    def set_config(self,
                   PosTaggerTool=None,
                   NerTaggerTool=None,
                   SpoTaggerTool=None,
                   SrlTaggerTool=None,
                   EventTaggerTool=None,
                   WikipediaLinkerTool=None,
                   WikipediaSearcherTool=None,
                   WikipediaEmbedderTool=None,
                   WikipediaSearcherPath=None,
                   WikipediaEmbedderPath=None):
        if PosTaggerTool is not None:
            self.tool_config["PosTagger"] = PosTaggerTool
        if NerTaggerTool is not None:
            self.tool_config["NerTagger"] = NerTaggerTool
        if SpoTaggerTool is not None:
            self.tool_config["SpoTagger"] = SpoTaggerTool
        if SrlTaggerTool is not None:
            self.tool_config["SrlTagger"] = SrlTaggerTool
        if EventTaggerTool is not None:
            self.tool_config["EventTagger"] = EventTaggerTool
        if WikipediaLinkerTool is not None:
            self.tool_config["WikipediaLinker"] = WikipediaLinkerTool
        if WikipediaSearcherTool is not None:
            self.tool_config["WikipediaSearcher"] = WikipediaSearcherTool
        if WikipediaEmbedderTool is not None:
            self.tool_config["WikipediaEmbedder"] = WikipediaEmbedderTool
        if WikipediaSearcherPath is not None:
            self.path_config["WikipediaSearcher"] = WikipediaSearcherPath
        if WikipediaEmbedderPath is not None:
            self.path_config["WikipediaEmbedder"] = WikipediaEmbedderPath

    def get_knowledge(self, sentence):
        knowledge_dict = {}
        link_list = []
        if self.return_pos:
            knowledge_dict["pos"] = self.pos_tagger.tag(sentence)
        if self.return_ner:
            knowledge_dict["ner"] = self.ner_tagger.tag(sentence)
        if self.return_srl:
            knowledge_dict["srl"] = self.srl_tagger.tag(sentence)
        if self.return_entity_desc or self.return_entity_ebd:
            link_list = self.wikipedia_linker.link(sentence)
        if self.return_entity_desc:
            for i, entity in enumerate(link_list):
                link_list[i]["desc"] = self.wikipedia_searcher.search(entity["id"])
        if self.return_entity_ebd:
            for i, entity in enumerate(link_list):
                link_list[i]["ebd"] = self.wikipedia_embedder.embed(entity["title"])
        knowledge_dict["wikipedia"] = link_list

        return knowledge_dict

    def info(self):
        # console = Console()
        # example_dict = self.get_knowledge(sentence="Bert likes reading in the Sesame Street Library.")
        # console.log(example_dict, log_locals=True)
        pass

    def help_tagger(self):
        self._help(TABLE=TABLE_DATA_TAGGER)

    def help_linker(self):
        self._help(TABLE=TABLE_DATA_LINKER)

    def help_searcher(self):
        self._help(TABLE=TABLE_DATA_SEARCHER)

    def help_embedder(self):
        self._help(TABLE=TABLE_DATA_EMBEDDER)

    def help(self):
        self.help_tagger()
        self.help_linker()
        self.help_searcher()
        self.help_embedder()

    def _help(self, TABLE):
        console = Console()

        table = Table(show_header=True)
        table.title = ("[not italic]:book:[/] CogKTR Enhancer Library [not italic]:book:[/]")
        table.add_column("Enhance Module")
        table.add_column("Enhance Function")
        table.add_column("Tool")
        table.add_column("Return Knowledge")
        table.columns[0].justify = "left"
        table.columns[1].justify = "left"
        table.columns[2].justify = "left"
        table.columns[3].justify = "left"
        table.columns[0].style = "red"
        table.columns[0].header_style = "bold red"
        table.columns[1].style = "blue"
        table.columns[1].header_style = "bold blue"
        table.columns[2].style = "green"
        table.columns[2].header_style = "bold green"
        table.columns[3].style = "cyan"
        table.columns[3].header_style = "bold cyan"
        table.row_styles = ["none", "dim"]
        table.border_style = "bright_yellow"
        table.width = console.width
        for row in TABLE:
            table.add_row(*row)
        console.print(table)


if __name__ == "__main__":
    from cogktr import *

    enhancer = Enhancer(return_pos=True,
                        return_ner=True,
                        return_spo=True,
                        return_srl=True,
                        return_event=True,
                        return_entity_desc=True,
                        return_entity_ebd=True)
    enhancer.help()
    enhancer.set_config(
        WikipediaSearcherPath="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia/raw_data/entity.jsonl",
        WikipediaEmbedderPath="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl")
    enhancer.info()
    sentence = "Bert likes reading in the Sesame Street Library."
    knowledge_dict = enhancer.get_knowledge(sentence=sentence)

    print("end")
