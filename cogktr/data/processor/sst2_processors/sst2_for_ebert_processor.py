import sys

import torch
from transformers import BertTokenizer
from cogkge.data.processor import BaseProcessor
from transformers import BertModel

from cogktr import DataTable, DataTableSet


class Sst2ForEbertProcessor(BaseProcessor):
    def __init__(self, plm, device, matrix_load_path):
        super().__init__()
        self.plm = plm
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_word_embedding = self.bert.get_input_embeddings().weight
        self.device = device
        self.matrix_load_path = matrix_load_path

    def _process(self, data, enhanced_dict):
        sentences = data.datas["sentence"]

        sentences_num = len(sentences)
        datable = DataTable()

        for idx, sentence in enumerate(sentences):
            if idx % 10000 == 0:
                print("Progress of process: {}/{}".format(idx, sentences_num))
                sys.stdout.flush()
            entity_list = enhanced_dict[sentence]
            label = int(data.datas["label"][idx])
            sentence_piece_list = []
            entity_mark_list = []
            curr_loc = 0

            # split sentence into list based on entities.
            # "Bert likes reading in the Sesame Street Library." ->
            # parted_sentence_list: ["Bert", " likes reading in the ", "Sesame Street", " Library."]
            # entity_mark_list: [1, 0, 1, 0]
            for entity in entity_list:
                start_loc = entity["start_loc"]
                end_loc = entity["end_loc"]

                sentence_piece = sentence[curr_loc:start_loc]
                if len(sentence_piece) != 0:
                    sentence_piece_list.append(sentence_piece)
                    entity_mark_list.append(0)

                sentence_piece = sentence[start_loc:end_loc]
                sentence_piece_list.append(sentence_piece)
                entity_mark_list.append(1)
                curr_loc = end_loc

            sentence_piece = sentence[curr_loc:]
            if len(sentence_piece) != 0:
                sentence_piece_list.append(sentence_piece)
                entity_mark_list.append(0)

            # tokenizer and get sentence embedding.
            entity_no = 0
            embeddings_list = []
            for (sentence_piece, entity_mark) in zip(sentence_piece_list, entity_mark_list):
                if entity_mark == 0:
                    tokens = self.tokenizer.tokenize(sentence_piece)
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    embeddings = self.bert_word_embedding[token_ids].to(self.device) # lens * 768

                else:
                    wiki_embedding = torch.FloatTensor(enhanced_dict[sentence]["wikipedia"][entity_no]["embedding"],
                                                       device = self.device) # 1 * 100
                    mapping_matrix = torch.load(self.matrix_load_path).to(self.device) # 100 * 768
                    embeddings = wiki_embedding * mapping_matrix # 1 * 768
                    entity_no = entity_no + 1

                embeddings_list.append(embeddings)

            input_embeds = torch.cat(tuple(embeddings_list), 0)

            datable("input_embeds", input_embeds)
            datable("label", label)

        return DataTableSet(datable)

    def process_train(self, data, enhanced_dict):
        return self._process(data, enhanced_dict)

    def process_dev(self, data, enhanced_dict):
        return self._process(data, enhanced_dict)

    def process_test(self, data, enhanced_dict):
        sentences = data.datas["sentence"]

        sentences_num = len(sentences)
        datable = DataTable()

        for idx, sentence in enumerate(sentences):
            if idx % 10000 == 0:
                print("Progress of process: {}/{}".format(idx, sentences_num))
                sys.stdout.flush()
            entity_list = enhanced_dict[sentence]
            sentence_piece_list = []
            entity_mark_list = []
            curr_loc = 0

            # split sentence into list based on entities.
            # "Bert likes reading in the Sesame Street Library." ->
            # parted_sentence_list: ["Bert", " likes reading in the ", "Sesame Street", " Library."]
            # entity_mark_list: [1, 0, 1, 0]
            for entity in entity_list:
                start_loc = entity["start_loc"]
                end_loc = entity["end_loc"]

                sentence_piece = sentence[curr_loc:start_loc]
                if len(sentence_piece) != 0:
                    sentence_piece_list.append(sentence_piece)
                    entity_mark_list.append(0)

                sentence_piece = sentence[start_loc:end_loc]
                sentence_piece_list.append(sentence_piece)
                entity_mark_list.append(1)
                curr_loc = end_loc

            sentence_piece = sentence[curr_loc:]
            if len(sentence_piece) != 0:
                sentence_piece_list.append(sentence_piece)
                entity_mark_list.append(0)

            # tokenizer and get sentence embedding.
            entity_no = 0
            embeddings_list = []
            for (sentence_piece, entity_mark) in zip(sentence_piece_list, entity_mark_list):
                if entity_mark == 0:
                    tokens = self.tokenizer.tokenize(sentence_piece)
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    embeddings = self.bert_word_embedding[token_ids].to(self.device)  # lens * 768

                else:
                    wiki_embedding = torch.FloatTensor(enhanced_dict[sentence]["wikipedia"][entity_no]["embedding"],
                                                       device=self.device)  # 1 * 100
                    mapping_matrix = torch.load(self.matrix_load_path).to(self.device)  # 100 * 768
                    embeddings = wiki_embedding * mapping_matrix  # 1 * 768
                    entity_no = entity_no + 1

                embeddings_list.append(embeddings)

            input_embeds = torch.cat(tuple(embeddings_list), 0)

            datable("input_embeds", input_embeds)

        return DataTableSet(datable)

