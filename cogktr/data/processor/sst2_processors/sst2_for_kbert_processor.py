# coding: utf-8
"""
KnowledgeGraph
"""
import os
import sys
import numpy as np
from cogktr.data.processor.base_processor import BaseProcessor
from cogktr.utils.constant.kbert_constants.constants import *
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer

class KnowledgeGraph(object):
    def __init__(self, spo_file_paths, predicate=True):
        self.predicate = predicate
        self.spo_file_paths = spo_file_paths
        self.lookup_table = self._create_lookup_table()
        # self.segment_vocab = list(self.lookup_table.keys()) + NEVER_SPLIT_TAG
        # self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.special_tags = set(NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred + " " + obje
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=2, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        # split_sent_batch = [nltk.word_tokenize(sent) for sent in sent_batch]
        split_sent_batch = [self.tokenizer.tokenize(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:

                entities = list(self.lookup_table.get(token, []))[:max_entities]
                sent_tree.append((token, entities))

                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(self.tokenizer.tokenize(ent)) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(self.tokenizer.tokenize(ent)) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = [word]
                    know_sent += add_word
                    seg += [0]
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = self.tokenizer.tokenize(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src
                    visible_matrix[id, visible_abs_idx] = 1
                    visible_matrix[visible_abs_idx, id] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1
                        visible_matrix[visible_abs_idx, id] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch

class Sst2ForKbertProcessor(BaseProcessor):
    def __init__(self, plm, spo_file_paths, max_token_len):
        super().__init__()
        self.plm = plm
        self.tokenizer = BertTokenizer.from_pretrained(plm)
        self.kg = KnowledgeGraph(spo_file_paths=spo_file_paths, predicate=True)
        self.max_token_len = max_token_len

    def _process(self, dataset):

        sentences = dataset.datas["sentence"]

        sentences_num = len(sentences)
        datable = DataTable()

        for sentence_id, sentence in enumerate(sentences):
            if sentence_id % 10000 == 0:
                print("Progress of process: {}/{}".format(sentence_id, sentences_num))
                sys.stdout.flush()
            try:
                label = dataset.datas["label"][sentence_id]
                text = CLS_TOKEN + sentence

                tokens, pos, vm, _ = self.kg.add_knowledge_with_vm([text], add_pad=True, max_length=self.max_token_len)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0]

                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                datable("token_ids", token_ids)
                datable("label", label)
                datable("mask", mask)
                datable("pos", pos)
                datable("vm", vm)
            except:
                print("Error line: ", sentence_id)

        return DataTableSet(datable)

    def process_train(self, train_data):
        return self._process(train_data)

    def process_dev(self, dev_data):
        return self._process(dev_data)

    def process_test(self, test_data):
        sentences = test_data.datas["sentence"]

        sentences_num = len(sentences)
        datable = DataTable()

        for sentence_id, sentence in enumerate(sentences):
            if sentence_id % 10000 == 0:
                print("Progress of process: {}/{}".format(sentence_id, sentences_num))
                sys.stdout.flush()
            try:
                text = CLS_TOKEN + sentence

                tokens, pos, vm, _ = self.kg.add_knowledge_with_vm([text], add_pad=True, max_length=self.max_token_len)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0]

                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                datable("token_ids", token_ids)
                datable("mask", mask)
                datable("pos", pos)
                datable("vm", vm)
            except:
                print("Error line: ", sentence_id)

        return DataTableSet(datable)

if __name__ == "__main__":
    from cogktr.data.reader.sst2_reader import Sst2Reader

    reader = Sst2Reader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()

    # kg = KnowledgeGraph(spo_file_paths=["/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikidata/wikidata.spo"],
    #                     predicate=True)
    # know_sent_batch, position_batch, visible_matrix_batch, seg_batch = kg.add_knowledge_with_vm(sent_batch=["hide new secretions from the parental units",
    #                                      "contains no wit , only labored gags",
    #                                      "that loves its characters and communicates something rather beautiful about human nature",
    #                                      "for those moviegoers who complain that ` they do n't make movies like they used to anymore",
    #                                      "equals the original and in some ways even betters it"])
    # know_sent_batch, position_batch, visible_matrix_batch, seg_batch = kg.add_knowledge_with_vm(
    #     sent_batch=["Tim Cook is visiting Beijing now."])
    processor = Sst2ForKbertProcessor(plm="bert-base-uncased", \
                                      spo_file_paths=["/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikidata/wikidata.spo"], \
                                      max_token_len=256)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")

