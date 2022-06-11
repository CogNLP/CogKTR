from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from tqdm import tqdm
from transformers import BertTokenizer
import transformers
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class Conll2003Processor(BaseProcessor):
    def __init__(self, plm, max_token_len, max_label_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.max_label_len = max_label_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for words, ner_labels in tqdm(zip(data['sentence'], data['ner_labels']), total=len(data['sentence'])):
            input_tokens = []
            input_ids = []
            attention_masks = []
            segment_ids = []
            valid_masks = []
            label_ids = []
            label_masks = []

            words_len = len(words)
            words.insert(0, '[CLS]')
            words.append('[SEP]')
            ner_labels.insert(0, 'O')
            ner_labels.append('O')
            for word in words:
                token = self.tokenizer.tokenize(word)
                input_tokens.extend(token)
                for i in range(len(token)):
                    valid_masks.append(1 if i == 0 else 0)
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            attention_masks = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            for label in ner_labels:
                label_ids.append(self.vocab["ner_label_vocab"].label2id(label))
            label_masks = [0] + [1] * (len(label_ids) - 2) + [0]

            input_ids = input_ids[0:self.max_token_len]
            attention_masks = attention_masks[0:self.max_token_len]
            segment_ids = segment_ids[0:self.max_token_len]
            valid_masks = valid_masks[0:self.max_token_len]
            label_ids = label_ids[0:self.max_label_len]
            label_masks = label_masks[0:self.max_label_len]

            input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
            attention_masks += [0 for _ in range(self.max_token_len - len(attention_masks))]
            segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
            valid_masks += [0 for _ in range(self.max_token_len - len(valid_masks))]
            label_ids += [-1 for _ in range(self.max_label_len - len(label_ids))]
            label_masks += [0 for _ in range(self.max_label_len - len(label_masks))]

            datable("input_ids", input_ids)
            datable("attention_masks", attention_masks)
            datable("segment_ids", segment_ids)
            datable("valid_masks", valid_masks)
            datable("label_ids", label_ids)
            datable("label_masks", label_masks)
            datable("words_len", words_len)

        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)


if __name__ == "__main__":
    from cogktr.data.reader.conll2003_reader import Conll2003Reader

    reader = Conll2003Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2003/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = Conll2003Processor(plm="bert-base-cased", max_token_len=256, max_label_len=256, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
