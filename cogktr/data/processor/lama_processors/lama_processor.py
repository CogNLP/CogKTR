from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class LamaProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence, object in tqdm(zip(data['masked_sent'], data['object']), total=len(data['masked_sent'])):
            token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
                                          max_length=self.max_token_len)
            masked_index = token.index(self.tokenizer.mask_token_id)
            labels = [-100] * self.max_token_len
            labels[masked_index] = self.tokenizer.convert_tokens_to_ids(object)
            masked_token_id = self.tokenizer.convert_tokens_to_ids(object)
            datable("input_ids", token)
            datable("masked_index", masked_index)
            datable("masked_token_id", masked_token_id)
            datable("labels", labels)
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)


if __name__ == "__main__":
    from cogktr.data.reader.lama_reader import LamaReader

    reader = LamaReader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/question_answering/LAMA/raw_data")
    train_data, dev_data, test_data = reader.read_all(dataset_name="Google_RE")
    vocab = reader.read_vocab()

    processor = LamaProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
