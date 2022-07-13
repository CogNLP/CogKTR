import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, OrderedDict

SEMCOR2WN_POS_TAG = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV}
STOPWORDS = list(set(stopwords.words('english')))


class TSemcorReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_semcor_raw_data(raw_data_path)
        self.train_file = 'Evaluation_Datasets/semeval2007/semeval2007.data.xml'
        self.train_gold_file = 'Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
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
        self.addition = {}
        self.addition["train"] = {}
        self.addition["train"]["instance_loc"] = {}
        self.addition["train"]["instance_label"] = {}
        self.addition["dev"] = {}
        self.addition["dev"]["instance_loc"] = {}
        self.addition["dev"]["instance_label"] = {}
        self.addition["test"] = {}
        self.addition["test"]["instance_loc"] = {}
        self.addition["test"]["instance_label"] = {}

    def _read_data(self, path, gold_path=None, datatype=None):
        datable = DataTable()
        print("Reading data...")

        # read data
        root = ET.parse(path).getroot()
        sentences = root.findall('.//sentence')
        for sentence in sentences:
            word_list = []
            tag_list = []
            lemma_list = []
            pos_list = []
            for index, word in enumerate(sentence):
                word_list.append(word.text)
                tag_list.append(word.tag)
                lemma_list.append(word.get("pos"))
                pos_list.append(word.get("lemma"))
                if word.tag == "instance":
                    self.addition[datatype]["instance_loc"][word.get("id")] = index
            datable("word_list", word_list)
            datable("tag_list", tag_list)
            datable("lemma_list", lemma_list)
            datable("pos_list", pos_list)
            datable("sentence_id", sentence.get("id"))

        # raed gold
        with open(gold_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split()
                key = line[0]
                # regard the first lemma label as gold label
                label = line[1]
                self.addition[datatype]["instance_label"][key] = label

        # read instance
        sample_list = []
        gloss_dict = {}
        instances = root.findall('.//instance')
        for instance in instances:
            instance_id = instance.get("id")
            gold_lemma_label = self.addition[datatype]["instance_label"][instance_id]
            sample_list.append((instance_id, gold_lemma_label, 1))
            instance_pos = SEMCOR2WN_POS_TAG[instance.get("pos")]
            # instance_lemma:'refer',represent word lemma
            instance_lemma = instance.get("lemma")
            # search lemma candidate set in WordNet according to pos tag
            # input:instance_lemma:"refer",instance_pos:"v"
            # return:Lemma('mention.v.01.refer'),Lemma('refer.v.02.refer')...
            # word.pos.id.word_item_name
            wn_lemmas = wn.lemmas(instance_lemma, instance_pos)
            for wn_lemma in wn_lemmas:
                # wn_lemma_key:'refer%2:32:01::',represnet lemma order number
                wn_lemma_key = wn_lemma.key()
                # wn_lemma_synset:Synset('mention.v.01'),represent lemma synset
                wn_lemma_synset = wn_lemma.synset()
                # wn_lemma_definition:'make reference to',represnt synset definition
                wn_lemma_definition = wn_lemma_synset.definition()
                # synonym
                # wn_lemma_synset_lemma:Lemma('mention.v.01.mention'),Lemma('mention.v.01.advert'),Lemma('mention.v.01.bring_up')...
                wn_lemma_synset_lemma = wn_lemma_synset.lemmas()
                synonym_words = []
                for lemma_lemma in wn_lemma_synset_lemma:
                    synonym_words.append(lemma_lemma)
                # example
                wn_lemma_synset_example = wn_lemma_synset.examples()
                examples_list = []
                for lemma_example in wn_lemma_synset_example:
                    examples_list.append(lemma_example)
                # hypernym
                wn_lemma_synset_hypernym = wn_lemma_synset.hypernyms()
                hypernym_list = []
                for lemma_hypernym in wn_lemma_synset_hypernym:
                    hypernym_list.append(lemma_hypernym.definition())
                gloss_dict[wn_lemma_key] = {}
                gloss_dict[wn_lemma_key]["lemma_definition"] = wn_lemma_definition
                gloss_dict[wn_lemma_key]["synonym"] = synonym_words
                gloss_dict[wn_lemma_key]["example"] = examples_list
                gloss_dict[wn_lemma_key]["hypernym"] = hypernym_list
                if wn_lemma_key != gold_lemma_label:
                    sample_list.append((instance_id, wn_lemma_key, 0))

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
