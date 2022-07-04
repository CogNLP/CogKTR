from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import AutoTokenizer
from tqdm import tqdm
from cogktr.data.processor.base_processor import BaseProcessor
import transformers
import numpy as np

transformers.logging.set_verbosity_error()  # set transformers logging level


class CommonsenseqaQagnnProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = AutoTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for context_list, candidate_text_list, answer_label, example_id in tqdm(
                zip(data['context'], data['candidate_text_list'], data['answer_label'], data['example_id']),
                total=len(data['example_id'])):
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            special_tokens_mask_list = []
            for context, candidate_text in zip(context_list, candidate_text_list):
                # TODO:check if need add " " before candidate text
                tokenized_data = self.tokenizer.encode_plus(text=context, text_pair=candidate_text,
                                                            truncation='longest_first',
                                                            padding="max_length",
                                                            add_special_tokens=True,
                                                            return_token_type_ids=True,
                                                            return_special_tokens_mask=True,
                                                            max_length=self.max_token_len)
                input_ids_list.append(tokenized_data['input_ids'])
                attention_mask_list.append(tokenized_data['attention_mask'])
                token_type_ids_list.append(tokenized_data['token_type_ids'])
                special_tokens_mask_list.append(tokenized_data['special_tokens_mask'])
            datable("example_id", example_id)
            datable("input_ids", np.array(input_ids_list))
            datable("attention_mask", np.array(attention_mask_list))
            datable("token_type_ids_list", np.array(token_type_ids_list))
            datable("special_tokens_mask_list", np.array(special_tokens_mask_list))
            datable("label", self.vocab["label_vocab"].label2id(answer_label))
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        datable = DataTable()
        print("Processing data...")
        for context_list, candidate_text_list, example_id in tqdm(
                zip(data['context'], data['candidate_text_list'], data['example_id']),
                total=len(data['example_id'])):
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            special_tokens_mask_list = []
            for context, candidate_text in zip(context_list, candidate_text_list):
                # TODO:check if need add " " before candidate text
                tokenized_data = self.tokenizer.encode_plus(text=context, text_pair=candidate_text,
                                                            truncation='longest_first',
                                                            padding="max_length",
                                                            add_special_tokens=True,
                                                            return_token_type_ids=True,
                                                            return_special_tokens_mask=True,
                                                            max_length=self.max_token_len)
                input_ids_list.append(tokenized_data['input_ids'])
                attention_mask_list.append(tokenized_data['attention_mask'])
                token_type_ids_list.append(tokenized_data['token_type_ids'])
                special_tokens_mask_list.append(tokenized_data['special_tokens_mask'])
            datable("example_id", example_id)
            datable("input_ids", np.array(input_ids_list))
            datable("attention_mask", np.array(attention_mask_list))
            datable("token_type_ids_list", np.array(token_type_ids_list))
            datable("special_tokens_mask_list", np.array(special_tokens_mask_list))
        return DataTableSet(datable)


if __name__ == "__main__":
    from cogktr.data.reader.commonsenseqa_qagnn_reader import CommonsenseqaQagnnReader

    reader = CommonsenseqaQagnnReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/question_answering/CommonsenseQA_for_QAGNN/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    addition = reader.read_addition()

    processor = CommonsenseqaQagnnProcessor(plm="roberta-large", max_token_len=100, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
