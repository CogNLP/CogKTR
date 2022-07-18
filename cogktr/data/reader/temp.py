import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from collections import defaultdict


class TSemcorReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_semcor_raw_data(raw_data_path)
        self.train_file = 'Training_Corpora/SemCor/semcor.data.xml'
        self.train_gold_file = 'Training_Corpora/SemCor/semcor.gold.key.txt'
        self.dev_file = 'Evaluation_Datasets/semeval2007/semeval2007.data.xml'
        self.dev_gold_file = 'Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
        self.test_file = 'Evaluation_Datasets/semeval2007/semeval2007.data.xml'
        self.test_gold_file = 'Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.train_gold_path = os.path.join(raw_data_path, self.train_gold_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.dev_gold_path = os.path.join(raw_data_path, self.dev_gold_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.test_gold_path = os.path.join(raw_data_path, self.test_gold_file)
        self.label_vocab = Vocabulary()
        self.label_vocab.add_dict({0: 0, 1: 1})
        self.addition = {}
        self.addition["train"] = {}
        self.addition["train"]["instance"] = defaultdict(dict)
        self.addition["train"]["sentence"] = defaultdict(dict)
        self.addition["train"]["example"] = []
        self.addition["train"]["segmentation"] = []
        self.addition["dev"] = {}
        self.addition["dev"]["instance"] = defaultdict(dict)
        self.addition["dev"]["sentence"] = defaultdict(dict)
        self.addition["dev"]["example"] = []
        self.addition["dev"]["segmentation"] = []
        self.addition["test"] = {}
        self.addition["test"]["instance"] = defaultdict(dict)
        self.addition["test"]["sentence"] = defaultdict(dict)
        self.addition["test"]["example"] = []
        self.addition["test"]["segmentation"] = []

        self.SEMCOR2WN_POS_TAG = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV}

    def _read_data(self, path, gold_path=None, datatype=None):
        datable = DataTable()

        # read data
        root = ET.parse(path).getroot()
        sentences = root.findall('.//sentence')
        for sentence in sentences:
            word_list = []
            tag_list = []
            lemma_list = []
            pos_list = []
            instance_list = []
            instance_pos_list = []
            instance_id_list = []
            for index, word in enumerate(sentence):
                word_list.append(word.text)
                tag_list.append(word.tag)
                lemma_list.append(word.get("lemma"))
                pos_list.append(word.get("pos"))
                if word.tag == "instance":
                    self.addition[datatype]["instance"][word.get("id")]["instance_loc"] = index
                    instance_list.append(word.get("lemma"))
                    instance_pos_list.append(self.SEMCOR2WN_POS_TAG[word.get("pos")])
                    instance_id_list.append(word.get("id"))
            self.addition[datatype]["sentence"][sentence.get("id")] = word_list
            datable("sentence_id", sentence.get("id"))
            datable("sentence", word_list)
            datable("tag_list", tag_list)
            datable("lemma_list", lemma_list)
            datable("pos_list", pos_list)
            datable("instance_id_list", instance_id_list)
            datable("instance_list", instance_list)
            datable("instance_pos_list", instance_pos_list)

        # raed gold
        with open(gold_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split()
                instance_id = line[0]
                # regard the first lemma label as gold label
                label = line[1]
                self.addition[datatype]["instance"][instance_id]["instance_label"] = label

        # read instance
        start = 0
        self.addition[datatype]["segmentation"].append(0)
        instances = root.findall('.//instance')
        for instance in instances:
            word_lemma_items = wn.lemmas(instance.get("lemma"), self.SEMCOR2WN_POS_TAG[instance.get("pos")])
            word_lemma_keys = [word_lemma_item.key() for word_lemma_item in word_lemma_items]
            gold_word_lemma_key = self.addition[datatype]["instance"][instance.get("id")]["instance_label"]
            for word_lemma_key, word_lemma_item in zip(word_lemma_keys, word_lemma_items):
                if gold_word_lemma_key == word_lemma_key:
                    self.addition[datatype]["example"].append(
                        (instance.get("id"), word_lemma_key, word_lemma_item._synset._definition, 1))
                else:
                    self.addition[datatype]["example"].append(
                        (instance.get("id"), word_lemma_key, word_lemma_item._synset._definition, 0))
            start += len(word_lemma_keys)
            self.addition[datatype]["segmentation"].append(start)
        return datable

    def _read_train(self, path, gold_path=None, datatype="train"):
        return self._read_data(path, gold_path, datatype)

    def _read_dev(self, path, gold_path=None, datatype="dev"):
        return self._read_data(path, gold_path, datatype)

    def _read_test(self, path, gold_path=None, datatype="test"):
        return self._read_data(path, gold_path, datatype)

    def read_all(self):
        return self._read_train(self.train_path, self.train_gold_path), \
               self._read_dev(self.dev_path, self.dev_gold_path), \
               self._read_test(self.test_path, self.test_gold_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}

    def read_addition(self):
        return self.addition


if __name__ == "__main__":
    reader = TSemcorReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    addition = reader.read_addition()
    print("end")
