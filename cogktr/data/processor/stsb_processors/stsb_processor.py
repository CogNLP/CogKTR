from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
from cogktr.data.processor.base_processor import BaseProcessor
import transformers

transformers.logging.set_verbosity_error()  # set transformers logging level


class StsbProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence1, sentence2, score in tqdm(zip(data['sentence1'], data['sentence2'], data['score']),
                                                total=len(data['score'])):
            tokenized_data = self.tokenizer.encode_plus(text=sentence1, text_pair=sentence2,
                                                        truncation='longest_first',
                                                        padding="max_length",
                                                        add_special_tokens=True,
                                                        max_length=self.max_token_len)
            datable("input_ids", tokenized_data["input_ids"])
            datable("token_type_ids", tokenized_data["token_type_ids"])
            datable("attention_mask", tokenized_data["attention_mask"])
            datable("label", float(score))
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence1, sentence2 in tqdm(zip(data['sentence1'], data['sentence2']), total=len(data['sentence1'])):
            tokenized_data = self.tokenizer.encode_plus(text=sentence1, text_pair=sentence2,
                                                        truncation='longest_first',
                                                        padding="max_length",
                                                        add_special_tokens=True,
                                                        max_length=self.max_token_len)
            datable("input_ids", tokenized_data["input_ids"])
            datable("token_type_ids", tokenized_data["token_type_ids"])
            datable("attention_mask", tokenized_data["attention_mask"])
        return DataTableSet(datable)


if __name__ == "__main__":
    from cogktr.data.reader.stsb_reader import StsbReader

    reader = StsbReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/STS_B/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = StsbProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
