from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class LamaProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, debug=False):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.debug = debug
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data):
        data = self.debug_process(data)
        datable = DataTable()
        print("Processing data...")
        for sentence, object in tqdm(zip(data['masked_sent'], data['object']), total=len(data['masked_sent'])):
            token = self.tokenizer.encode(text=sentence,
                                          truncation=True,
                                          padding="max_length",
                                          add_special_tokens=True,
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

    def process_test(self, data):
        return self._process(data)


if __name__ == "__main__":
    from cogktr.data.reader.lama_reader import LamaReader

    reader = LamaReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/masked_language_model/LAMA/raw_data")
    test_data = reader.read_all(dataset_name="google_re", isSplit=False)
    vocab = reader.read_vocab()

    processor = LamaProcessor(plm="bert-base-cased", max_token_len=512, vocab=vocab)
    test_dataset = processor.process_test(test_data)
    print("end")
