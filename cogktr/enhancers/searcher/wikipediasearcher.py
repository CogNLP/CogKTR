from cogktr.enhancers.searcher import *
import json
from tqdm import tqdm


# TODO: add cogie toolkit
class WikipediaSearcher(BaseSearcher):
    def __init__(self, tool, path, return_desc=True):
        super().__init__()
        if tool not in ["blink"]:
            raise ValueError("Please set a tool!")
        self.tool = tool
        self.path = path
        self.return_desc = return_desc

        self.title2id = {}
        self.id2title = {}
        self.id2desc = {}
        with open(self.path, "r") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                entity = json.loads(line)
                if "idx" in entity:
                    split = entity["idx"].split("curid=")
                    if len(split) > 1:
                        wikipedia_id = int(split[-1].strip())
                    else:
                        wikipedia_id = entity["idx"].strip()

                self.title2id[entity["title"]] = wikipedia_id
                self.id2title[wikipedia_id] = entity["title"]
                self.id2desc[wikipedia_id] = entity["text"]

    # TODO:use sparql to search neighbor node and more info
    def search(self, id):
        search_dict = {}
        if self.tool == "blink":
            search_dict = self._blink_search(id)
        return search_dict

    def _blink_search(self, id):
        search_dict = {}
        search_dict[id] = {}
        search_dict[id]["desc"] = None
        if self.return_desc:
            search_dict[id]["desc"] = self.id2desc[id]
        return search_dict


if __name__ == "__main__":
    searcher = WikipediaSearcher(tool="blink",
                                 path="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia/raw_data/entity.jsonl")
    search_dict = searcher.search(18978754)
    print("end")
