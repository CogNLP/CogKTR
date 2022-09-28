from cogktr.enhancers.world_enhancer import WorldEnhancer
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
import numpy as np
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class Sst2ForKtattProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")

        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            words = enhanced_dict[sentence]["words"]
            input_tokens = []
            input_ids = []
            attention_mask = np.zeros((self.max_token_len, self.max_token_len))
            segment_ids = []
            valid_masks = []

            words_len = len(words)
            words.insert(0, '[CLS]')
            offset = {}
            for i, word in enumerate(words):
                token = self.tokenizer.tokenize(word)
                offset_start = len(input_tokens) if len(
                    input_tokens) <= self.max_token_len - 1 else self.max_token_len - 1
                offset_end = (len(input_tokens) + len(token)) if len(
                    input_tokens) <= self.max_token_len - 1 else self.max_token_len - 1
                offset[i] = (offset_start, offset_end)
                input_tokens.extend(token)
                for i in range(len(token)):
                    valid_masks.append(1 if i == 0 else 0)
            raw_sentence_token_len = len(input_tokens)
            attention_mask[0:raw_sentence_token_len, 0:raw_sentence_token_len] = 1

            if len(enhanced_dict[sentence]["entities"]) > 0:
                entity_num = len(enhanced_dict[sentence]["entities"])
                entity_desc_max_len = int((self.max_token_len - len(input_tokens)) / entity_num) - 1

            for entity in enhanced_dict[sentence]["entities"]:
                if entity["desc"] is not None:
                    entity_token = self.tokenizer.tokenize(text=entity["desc"])
                    entity_token = entity_token[:entity_desc_max_len]
                    row_start_loc = entity["start"] + 1
                    row_end_loc = entity["end"] + 2
                    column_start_loc = len(input_tokens)
                    input_tokens.extend(entity_token)
                    column_end_loc = len(input_tokens)
                    attention_mask[row_start_loc:row_end_loc, column_start_loc:column_end_loc] = 1
                    attention_mask[column_start_loc:column_end_loc, column_start_loc:column_end_loc] = 1

            input_tokens.append('[SEP]')
            attention_mask[len(input_tokens) - 1, :len(input_tokens)] = 1
            attention_mask[:len(input_tokens), len(input_tokens) - 1] = 1

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            segment_ids = [0] * len(input_ids)

            input_ids = input_ids[0:self.max_token_len]
            segment_ids = segment_ids[0:self.max_token_len]
            valid_masks = valid_masks[0:self.max_token_len]

            input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
            segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
            valid_masks += [0 for _ in range(self.max_token_len - len(valid_masks))]

            datable("input_ids", input_ids)
            datable("attention_mask", attention_mask)
            datable("segment_ids", segment_ids)
            datable("valid_masks", valid_masks)
            datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_train(self, data, enhanced_dict=None):
        return self._process(data, enhanced_dict)

    def process_dev(self, data, enhanced_dict=None):
        return self._process(data, enhanced_dict)

    def process_test(self, data, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")

        for sentence in tqdm(data['sentence'], total=len(data['sentence'])):
            words = enhanced_dict[sentence]["words"]
            input_tokens = []
            input_ids = []
            attention_mask = np.zeros((self.max_token_len, self.max_token_len))
            segment_ids = []
            valid_masks = []

            words_len = len(words)
            words.insert(0, '[CLS]')
            offset = {}
            for i, word in enumerate(words):
                token = self.tokenizer.tokenize(word)
                offset_start = len(input_tokens) if len(
                    input_tokens) <= self.max_token_len - 1 else self.max_token_len - 1
                offset_end = (len(input_tokens) + len(token)) if len(
                    input_tokens) <= self.max_token_len - 1 else self.max_token_len - 1
                offset[i] = (offset_start, offset_end)
                input_tokens.extend(token)
                for i in range(len(token)):
                    valid_masks.append(1 if i == 0 else 0)
            raw_sentence_token_len = len(input_tokens)
            attention_mask[0:raw_sentence_token_len, 0:raw_sentence_token_len] = 1

            if len(enhanced_dict[sentence]["entities"]) > 0:
                entity_num = len(enhanced_dict[sentence]["entities"])
                entity_desc_max_len = int((self.max_token_len - len(input_tokens)) / entity_num) - 1

            for entity in enhanced_dict[sentence]["entities"]:
                if entity["desc"] is not None:
                    entity_token = self.tokenizer.tokenize(text=entity["desc"])
                    entity_token = entity_token[:entity_desc_max_len]
                    row_start_loc = entity["start"] + 1
                    row_end_loc = entity["end"] + 2
                    column_start_loc = len(input_tokens)
                    input_tokens.extend(entity_token)
                    column_end_loc = len(input_tokens)
                    attention_mask[row_start_loc:row_end_loc, column_start_loc:column_end_loc] = 1
                    attention_mask[column_start_loc:column_end_loc, column_start_loc:column_end_loc] = 1

            input_tokens.append('[SEP]')
            attention_mask[len(input_tokens) - 1, :len(input_tokens)] = 1
            attention_mask[:len(input_tokens), len(input_tokens) - 1] = 1

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            segment_ids = [0] * len(input_ids)

            input_ids = input_ids[0:self.max_token_len]
            segment_ids = segment_ids[0:self.max_token_len]
            valid_masks = valid_masks[0:self.max_token_len]

            input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
            segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
            valid_masks += [0 for _ in range(self.max_token_len - len(valid_masks))]

            datable("input_ids", input_ids)
            datable("attention_mask", attention_mask)
            datable("segment_ids", segment_ids)
            datable("valid_masks", valid_masks)
        return DataTableSet(datable)


if __name__ == "__main__":
    from cogktr.data.reader.sst2_reader import Sst2Reader

    reader = Sst2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    enhancer = WorldEnhancer(knowledge_graph_path="/data/mentianyi/code/CogKTR/datapath/knowledge_graph",
                             cache_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/enhanced_data",
                             cache_file="world_knowledge",
                             reprocess=False,
                             load_entity_desc=True,
                             load_entity_embedding=False,
                             load_entity_kg=False)
    enhanced_train_dict = enhancer.enhance_train(datable=train_data,
                                                 enhanced_key_1="sentence",
                                                 return_entity_desc=True,
                                                 return_entity_embedding=False,
                                                 return_entity_kg=False)
    enhanced_dev_dict = enhancer.enhance_dev(datable=dev_data,
                                             enhanced_key_1="sentence",
                                             return_entity_desc=True,
                                             return_entity_embedding=False,
                                             return_entity_kg=False)
    enhanced_test_dict = enhancer.enhance_test(datable=test_data,
                                               enhanced_key_1="sentence",
                                               return_entity_desc=True,
                                               return_entity_embedding=False,
                                               return_entity_kg=False)

    processor = Sst2ForKtattProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
    train_dataset = processor.process_train(data=train_data, enhanced_dict=enhanced_train_dict)
    dev_dataset = processor.process_dev(data=dev_data, enhanced_dict=enhanced_dev_dict)
    test_dataset = processor.process_test(data=test_data, enhanced_dict=enhanced_test_dict)
    print("end")
