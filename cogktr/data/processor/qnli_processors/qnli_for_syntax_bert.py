from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class QnliForSyntaxBertProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def process_train(self, data, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")
        xx = 0
        for sentence, question, label in tqdm(zip(data['sentence'], data['question'], data['label']),
                                              total=len(data['sentence'])):
            if xx > 10:
                break
            tokenized_data = self.tokenizer.encode_plus(text=sentence, text_pair=question,
                                                        truncation='longest_first',
                                                        padding="max_length",
                                                        add_special_tokens=True,
                                                        max_length=self.max_token_len)
            datable("input_ids", tokenized_data["input_ids"])
            datable("token_type_ids", tokenized_data["token_type_ids"])
            datable("attention_mask", tokenized_data["attention_mask"])
            datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_dev(self, data, enhanced_dict=None):
        return self._process(data)

    def process_test(self, data, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")
        xx = 0
        for sentence, question in tqdm(zip(data['sentence'], data['question']),
                                       total=len(data['sentence'])):
            if xx > 10:
                break
            tokenized_data = self.tokenizer.encode_plus(text=sentence, text_pair=question,
                                                        truncation='longest_first',
                                                        padding="max_length",
                                                        add_special_tokens=True,
                                                        max_length=self.max_token_len)
            datable("input_ids", tokenized_data["input_ids"])
            datable("token_type_ids", tokenized_data["token_type_ids"])
            datable("attention_mask", tokenized_data["attention_mask"])
        return DataTableSet(datable)


if __name__ == "__main__":
    from cogktr.data.reader.qnli_reader import QnliReader

    reader = QnliReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = QnliForSyntaxBertProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
