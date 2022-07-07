from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import AutoTokenizer
from tqdm import tqdm
from cogktr.data.processor.base_processor import BaseProcessor
import transformers
import torch

transformers.logging.set_verbosity_error()  # set transformers logging level


class CommonsenseqaProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, mode):
        super().__init__()
        if mode not in ["replace", "concatenate"]:
            raise ValueError("{} in CommonsenseqaProcessor is not supported!".format(mode))

        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for question, statements, answerkey, example_id in tqdm(
                zip(data['question'], data['statements'], data['answerkey'], data['example_id']),
                total=len(data['example_id'])):
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            special_tokens_mask_list = []
            if self.mode == "replace":
                for statement in statements:
                    replace_statement = statement["statement"]
                    tokenized_data = self.tokenizer.encode_plus(text=replace_statement,
                                                                padding="max_length",
                                                                truncation=True,
                                                                add_special_tokens=True,
                                                                return_token_type_ids=True,
                                                                return_special_tokens_mask=True,
                                                                max_length=self.max_token_len)
                    input_ids_list.append(torch.tensor(tokenized_data['input_ids']))
                    attention_mask_list.append(torch.tensor(tokenized_data['attention_mask']))
                    token_type_ids_list.append(torch.tensor(tokenized_data['token_type_ids']))
                    special_tokens_mask_list.append(torch.tensor(tokenized_data['special_tokens_mask']))

            if self.mode == "concatenate":
                stem = question['stem']
                for choice in question['choices']:
                    tokenized_data = self.tokenizer.encode_plus(text=stem,
                                                                text_pair=choice["text"],
                                                                padding="max_length",
                                                                truncation=True,
                                                                add_special_tokens=True,
                                                                return_token_type_ids=True,
                                                                return_special_tokens_mask=True,
                                                                max_length=self.max_token_len)
                    input_ids_list.append(torch.tensor(tokenized_data['input_ids']))
                    attention_mask_list.append(torch.tensor(tokenized_data['attention_mask']))
                    token_type_ids_list.append(torch.tensor(tokenized_data['token_type_ids']))
                    special_tokens_mask_list.append(torch.tensor(tokenized_data['special_tokens_mask']))

            datable("input_ids", torch.stack(input_ids_list))
            datable("attention_mask", torch.stack(attention_mask_list))
            datable("token_type_ids", torch.stack(token_type_ids_list))
            datable("special_tokens_mask", torch.stack(special_tokens_mask_list))
            datable("label", self.vocab["label_vocab"].label2id(answerkey))
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data=data)

    def process_dev(self, data):
        return self._process(data=data)

    def process_test(self, data):
        datable = DataTable()
        print("Processing data...")
        for question, statements, example_id in tqdm(
                zip(data['question'], data['statements'], data['example_id']),
                total=len(data['example_id'])):
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            special_tokens_mask_list = []
            if self.mode == "replace":
                for statement in statements:
                    replace_statement = statement["statement"]
                    tokenized_data = self.tokenizer.encode_plus(text=replace_statement,
                                                                padding="max_length",
                                                                truncation=True,
                                                                add_special_tokens=True,
                                                                return_token_type_ids=True,
                                                                return_special_tokens_mask=True,
                                                                max_length=self.max_token_len)
                    input_ids_list.append(torch.tensor(tokenized_data['input_ids']))
                    attention_mask_list.append(torch.tensor(tokenized_data['attention_mask']))
                    token_type_ids_list.append(torch.tensor(tokenized_data['token_type_ids']))
                    special_tokens_mask_list.append(torch.tensor(tokenized_data['special_tokens_mask']))

            if self.mode == "concatenate":
                stem = question['stem']
                for choice in question['choices']:
                    tokenized_data = self.tokenizer.encode_plus(text=stem,
                                                                text_pair=choice["text"],
                                                                padding="max_length",
                                                                truncation=True,
                                                                add_special_tokens=True,
                                                                return_token_type_ids=True,
                                                                return_special_tokens_mask=True,
                                                                max_length=self.max_token_len)
                    input_ids_list.append(torch.tensor(tokenized_data['input_ids']))
                    attention_mask_list.append(torch.tensor(tokenized_data['attention_mask']))
                    token_type_ids_list.append(torch.tensor(tokenized_data['token_type_ids']))
                    special_tokens_mask_list.append(torch.tensor(tokenized_data['special_tokens_mask']))

            datable("input_ids", torch.stack(input_ids_list))
            datable("attention_mask", torch.stack(attention_mask_list))
            datable("token_type_ids", torch.stack(token_type_ids_list))
            datable("special_tokens_mask", torch.stack(special_tokens_mask_list))
        return DataTableSet(datable)


if __name__ == "__main__":
    from cogktr.data.reader.commonsenseqa_reader import CommonsenseqaReader

    reader = CommonsenseqaReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/question_answering/CommonsenseQA/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = CommonsenseqaProcessor(plm="bert-base-cased", max_token_len=100, vocab=vocab, mode="concatenate")
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
