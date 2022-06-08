from transformers import BertTokenizer


class SequenceBertTokenizer:
    def __init__(self, pretrained_model_name_or_path):
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        tokenizer = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path)
        return tokenizer

    def tokenize(self, word_sequence):
        tokenized_text = list()
        head_flag = list()
        for word in word_sequence:
            token_str = self._tokenizer.tokenize(word)
            tokenized_text += token_str
            head_flag += [1] + [0] * (len(token_str) - 1)
        return tokenized_text, head_flag

    def encode(
            self,
            word_sequence,
            max_length=None,
            add_special_tokens=True,
    ):
        if add_special_tokens:
            word_sequence = ["[CLS]"] + word_sequence + ["[SEP]"]
        input_ids = list()
        tokenized_text = list()
        attention_mask = list()
        head_flag_matrix = list()
        for word in word_sequence:
            token_str = self._tokenizer.tokenize(word)
            tokenized_text += token_str
            input_id = self._tokenizer.convert_tokens_to_ids(token_str)
            if token_str == ['[CLS]'] or token_str == ['[SEP]']:
                head_flag = [0] * max_length
            else:
                head_flag = [0] * len(input_ids) + [1] * len(input_id) + [0] * (
                        max_length - len(input_ids) - len(input_id))
                head_flag = head_flag[:max_length]

            input_ids += input_id
            attention_mask += [1] * len(input_id)
            head_flag_matrix.append(head_flag)

        input_ids = input_ids + max_length * [0]
        attention_mask = attention_mask + max_length * [0]

        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

        return input_ids, attention_mask, head_flag_matrix


if __name__ == "__main__":
    tokenizer = SequenceBertTokenizer.from_pretrained("bert-base-cased")
    word_sequence = "Bert of Sesame Street likes to go to the library to learn cognitive knowledge!".strip().split()
    tokenized_text, head_flag = tokenizer.tokenize(word_sequence=word_sequence)
    input_ids, attention_mask, head_flag_matrix = tokenizer.encode(word_sequence=word_sequence,
                                                                   max_length=128,
                                                                   add_special_tokens=True)
    print("end")
