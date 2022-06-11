from cogktr.data.reader.sst2_reader import Sst2Reader
from cogktr.enhancers import Enhancer
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class Sst2ForKtembProcessor(BaseProcessor):
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
            words = enhanced_dict[sentence]["wikipedia"]["words"]
            input_tokens = []
            input_ids = []
            attention_masks = []
            segment_ids = []
            valid_masks = []

            words_len = len(words)
            words.insert(0, '[CLS]')
            words.append('[SEP]')
            for word in words:
                token = self.tokenizer.tokenize(word)
                input_tokens.extend(token)
                for i in range(len(token)):
                    valid_masks.append(1 if i == 0 else 0)
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            attention_masks = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            input_ids = input_ids[0:self.max_token_len]
            attention_masks = attention_masks[0:self.max_token_len]
            segment_ids = segment_ids[0:self.max_token_len]
            valid_masks = valid_masks[0:self.max_token_len]

            input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
            attention_masks += [0 for _ in range(self.max_token_len - len(attention_masks))]
            segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
            valid_masks += [0 for _ in range(self.max_token_len - len(valid_masks))]

            entity_span_list = []
            for entity in enhanced_dict[sentence]["wikipedia"]["spans"]:
                if entity["desc"] is not None:
                    entity_token = self.tokenizer.encode(text=entity["desc"], truncation=True, padding="max_length",
                                                         add_special_tokens=True,
                                                         max_length=self.max_token_len)
                    entity_dict = {}
                    entity_dict["entity_token"] = entity_token
                    entity_dict["entity_mask"] = [0] * self.max_token_len
                    for i in range(entity["start"] + 1, entity["end"] + 1):  # because add [CLS]
                        entity_dict["entity_mask"][i] = 1
                    entity_span_list.append(entity_dict)
            datable("input_ids", input_ids)
            datable("attention_masks", attention_masks)
            datable("segment_ids", segment_ids)
            datable("valid_masks", valid_masks)
            datable("label", self.vocab["label_vocab"].label2id(label))
            datable("entity_span_list", entity_span_list)
        return DataTableSet(datable)

    def process_train(self, data, enhanced_dict=None):
        return self._process(data=data, enhanced_dict=enhanced_dict)

    def process_dev(self, data, enhanced_dict=None):
        return self._process(data=data, enhanced_dict=enhanced_dict)

    def process_test(self, data, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")
        for sentence in tqdm(zip(data['sentence']), total=len(data['sentence'])):
            words = enhanced_dict[sentence]["wikipedia"]["words"]
            input_tokens = []
            input_ids = []
            attention_masks = []
            segment_ids = []
            valid_masks = []

            words_len = len(words)
            words.insert(0, '[CLS]')
            words.append('[SEP]')
            for word in words:
                token = self.tokenizer.tokenize(word)
                input_tokens.extend(token)
                for i in range(len(token)):
                    valid_masks.append(1 if i == 0 else 0)
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            attention_masks = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            input_ids = input_ids[0:self.max_token_len]
            attention_masks = attention_masks[0:self.max_token_len]
            segment_ids = segment_ids[0:self.max_token_len]
            valid_masks = valid_masks[0:self.max_token_len]

            input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
            attention_masks += [0 for _ in range(self.max_token_len - len(attention_masks))]
            segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
            valid_masks += [0 for _ in range(self.max_token_len - len(valid_masks))]

            entity_span_list = []
            for entity in enhanced_dict[sentence]["wikipedia"]["spans"]:
                if entity["desc"] is not None:
                    entity_token = self.tokenizer.encode(text=entity["desc"], truncation=True, padding="max_length",
                                                         add_special_tokens=True,
                                                         max_length=self.max_token_len)
                    entity_dict = {}
                    entity_dict["entity_token"] = entity_token
                    entity_dict["entity_mask"] = [0] * self.max_token_len
                    for i in range(entity["start"] + 1, entity["end"] + 1):  # because add [CLS]
                        entity_dict["entity_mask"][i] = 1
                    entity_span_list.append(entity_dict)
            datable("input_ids", input_ids)
            datable("attention_masks", attention_masks)
            datable("segment_ids", segment_ids)
            datable("valid_masks", valid_masks)
            datable("entity_span_list", entity_span_list)
        return DataTableSet(datable)


if __name__ == "__main__":
    reader = Sst2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    enhancer = Enhancer(reprocess=False,
                        save_file_name="dev_enhance",
                        datapath="/data/mentianyi/code/CogKTR/datapath",
                        enhanced_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/enhanced_data")
    # enhanced_train_dict = enhancer.enhance_train(train_data)
    enhanced_dev_dict = enhancer.enhance_dev(dev_data)
    # enhanced_test_dict = enhancer.enhance_test(test_data)
    processor = Sst2ForKtembProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
    # train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(data=dev_data, enhanced_dict=enhanced_dev_dict)
    # test_dataset = processor.process_test(test_data)
    print("end")
