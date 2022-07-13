from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import AutoTokenizer
from tqdm import tqdm
import transformers
import torch
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class SemcorProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = AutoTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for input_ids_raw, input_len, context_len, instance, label in tqdm(
                zip(data['input_ids'], data['input_len'], data['context_len'], data['instance'], data['label']),
                total=len(data['input_ids'])):

            batch = {}
            input_ids = [0] * self.max_token_len
            attention_mask = [0] * self.max_token_len
            token_type_ids = [0] * self.max_token_len
            instance_mask = [0] * self.max_token_len
            input_ids[:input_len] =input_ids_raw
            attention_mask[:input_len] = [1] * input_len
            token_type_ids[context_len:input_len] = [1] * (input_len - context_len)
            instance_begin, instance_end, instance_lens = instance
            instance_mask[instance_begin:instance_end] = [1] * instance_lens
            datable("input_ids",input_ids)
            datable("attention_mask",attention_mask)
            # datable("token_type_ids",token_type_ids)
            datable("instance_mask",instance_mask)
            datable("instance_lens",instance_lens)
            datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)


if __name__ == "__main__":
    from cogktr.data.reader.semcor_reader import SemcorReader

    reader = SemcorReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = SemcorProcessor(plm="bert-base-cased", max_token_len=512, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
