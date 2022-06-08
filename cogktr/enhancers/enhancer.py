from rich.console import Console
from rich.table import Table
import numpy as np
import copy
import os
from tqdm import tqdm
from cogktr.utils.io_utils import load_json, save_json
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
                 return_entity_emb=False,
                 reprocess=True,
                 datapath=None,
                 enhanced_data_path=None):
        self.return_pos = return_pos
        self.return_ner = return_ner
        self.return_spo = return_spo
        self.return_srl = return_srl
        self.return_event = return_event
        self.return_entity_desc = return_entity_desc
        self.return_entity_emb = return_entity_emb
        self.reprocess = reprocess
        self.datapath = datapath
        self.enhanced_data_path = enhanced_data_path

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
        if self.return_entity_desc:
            self.path_config["WikipediaSearcher"] = os.path.join(self.datapath,
                                                                 "knowledge_graph/wikipedia/raw_data/entity.jsonl")
        if self.return_entity_emb:
            if self.tool_config["WikipediaEmbedder"] == "wikipedia2vec":
                self.path_config["WikipediaEmbedder"] = os.path.join(self.datapath,
                                                                     "knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl")
            if self.tool_config["WikipediaEmbedder"] == "cogkge":
                self.path_config["WikipediaEmbedder"]["model"] = os.path.join(self.datapath,
                                                                              "knowledge_graph/cogkge/raw_data/Model.pkl")
                self.path_config["WikipediaEmbedder"]["vocab"] = os.path.join(self.datapath,
                                                                              "knowledge_graph/cogkge/raw_data/vocab.pkl")

    def _init_module(self):
        if self.return_pos:
            self.pos_tagger = PosTagger(tool=self.tool_config["NerTagger"])
        if self.return_ner:
            self.ner_tagger = NerTagger(tool=self.tool_config["NerTagger"])
        if self.return_srl:
            self.srl_tagger = SrlTagger(tool=self.tool_config["SrlTagger"])
        if self.return_entity_desc or self.return_entity_emb:
            self.wikipedia_linker = WikipediaLinker(tool=self.tool_config["WikipediaLinker"])
        if self.return_entity_desc:
            self.wikipedia_searcher = WikipediaSearcher(tool=self.tool_config["WikipediaSearcher"],
                                                        path=self.path_config["WikipediaSearcher"])
        if self.return_entity_emb:
            if self.tool_config["WikipediaEmbedder"] == "wikipedia2vec":
                self.wikipedia_embedder = WikipediaEmbedder(tool=self.tool_config["WikipediaEmbedder"],
                                                            path=self.path_config["WikipediaEmbedder"])
            if self.tool_config["WikipediaEmbedder"] == "cogkge":
                self.wikipedia_embedder = WikipediaEmbedder(tool=self.tool_config["WikipediaEmbedder"],
                                                            path=self.path_config["WikipediaEmbedder"]["model"],
                                                            vocab_path=self.path_config["WikipediaEmbedder"]["vocab"])

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
        if self.return_entity_desc or self.return_entity_emb:
            link_list = self.wikipedia_linker.link(sentence)
        if self.return_entity_desc:
            for i, entity in enumerate(link_list):
                link_list[i]["desc"] = self.wikipedia_searcher.search(entity["id"])["desc"]
        link_list_copy = copy.deepcopy(link_list)
        if self.return_entity_emb:
            unaligned_num = 0
            for i, entity in enumerate(link_list_copy):
                list_point = i - unaligned_num
                entity_embedding = self.wikipedia_embedder.embed(entity["title"])["entity_embedding"]
                similar_entities = self.wikipedia_embedder.embed(entity["title"])["similar_entities"]
                if entity_embedding == np.array(100).tolist():
                    # Delete unaligned entity title
                    del link_list[list_point]
                    unaligned_num += 1
                else:
                    link_list[list_point]["entity_embedding"] = entity_embedding
                    link_list[list_point]["similar_entities"] = similar_entities

        knowledge_dict["wikipedia"] = link_list

        return knowledge_dict

    def _enhance(self, datable, enhanced_key_1="sentence", enhanced_key_2=None, dict_name=None):
        enhanced_dict = {}

        if not self.reprocess and os.path.exists(os.path.join(self.enhanced_data_path, dict_name)):
            enhanced_dict = load_json(os.path.join(self.enhanced_data_path, dict_name))
        else:
            print("Enhancing data...")
            if enhanced_key_2 is None:
                for sentence_1 in tqdm(datable[enhanced_key_1]):
                    enhanced_dict[sentence_1] = self.get_knowledge(sentence_1)
            if enhanced_key_2 is not None:
                for sentence_1, sentence_2 in tqdm(zip(datable[enhanced_key_1], datable[enhanced_key_2])):
                    enhanced_dict[sentence_1] = self.get_knowledge(sentence_1)
                    enhanced_dict[sentence_2] = self.get_knowledge(sentence_2)
            save_json(enhanced_dict, os.path.join(self.enhanced_data_path, dict_name))
        return enhanced_dict

    def enhance_train(self, datable, enhanced_key_1="sentence", enhanced_key_2=None):
        return self._enhance(datable=datable, enhanced_key_1=enhanced_key_1, enhanced_key_2=enhanced_key_2,
                             dict_name="enhanced_train_dict.json")

    def enhance_dev(self, datable, enhanced_key_1="sentence", enhanced_key_2=None):
        return self._enhance(datable=datable, enhanced_key_1=enhanced_key_1, enhanced_key_2=enhanced_key_2,
                             dict_name="enhanced_dev_dict.json")

    def enhance_test(self, datable, enhanced_key_1="sentence", enhanced_key_2=None):
        return self._enhance(datable=datable, enhanced_key_1=enhanced_key_1, enhanced_key_2=enhanced_key_2,
                             dict_name="enhanced_test_dict.json")

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

    reader = Sst2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    enhancer = Enhancer(return_pos=True,
                        return_ner=True,
                        return_spo=True,
                        return_srl=True,
                        return_event=True,
                        return_entity_desc=False,
                        return_entity_emb=True,
                        reprocess=True,
                        datapath="/data/mentianyi/code/CogKTR/datapath",
                        enhanced_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/enhanced_data")

    sentence = "Bert likes reading in the Sesame Street Library."
    knowledge_dict = enhancer.get_knowledge(sentence=sentence)

    # enhanced_train_dict = enhancer.enhance_train(train_data)
    enhanced_dev_dict = enhancer.enhance_dev(dev_data)
    # enhanced_test_dict = enhancer.enhance_test(test_data)

    print("end")
