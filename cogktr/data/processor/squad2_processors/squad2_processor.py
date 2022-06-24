from cogktr.data.reader.qnli_reader import QnliReader
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
from argparse import Namespace
from cogktr.utils.log_utils import logger
from tqdm import tqdm

transformers.logging.set_verbosity_error()  # set transformers logging level

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}

class Squad2Processor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, debug=False):
        super().__init__(debug=debug)
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        data = self.debug_process(data)
        for qas_id, is_impossible, \
            question_text, context_text, \
            answer_text, start_position, end_position, \
            doc_tokens, title \
                in tqdm(zip(data["qas_id"], data["is_impossible"],
                            data["question_text"], data["context_text"],
                            data["answer_text"], data["start_position"],
                            data["end_position"], data["doc_tokens"],
                            data["title"], ), total=len(data["qas_id"])):
            example = {
                "qas_id": qas_id,
                "is_impossible": is_impossible,
                "question_text": question_text,
                "context_text": context_text,
                "answer_text": answer_text,
                "start_position": start_position,
                "end_position": end_position,
                "doc_tokens": doc_tokens,
                "title": title,
            }
            example = Namespace(**example)
            result = process_bert(example,
                                  tokenizer=self.tokenizer,
                                  max_seq_length=self.max_token_len,
                                  doc_stride=128,
                                  max_query_length=64,
                                  padding_strategy="max_length",
                                  is_training=True)
            for key, value in result.items():
                datable(key, value)

        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)


def process_bert(example, tokenizer, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training):
    # start_position = example.start_position
    # end_position = example.end_position

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )


    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    paragraph_len = min(
        len(all_doc_tokens),
        max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
    )

    start_position = 0
    end_position = 0
    if is_training and not example.is_impossible:
        doc_start = 0
        doc_end = paragraph_len - 1
        out_of_span = False
        if not (tok_start_position >= start_position and tok_end_position <= doc_end):
            out_of_span = True

        if out_of_span:
            start_position = 0
            end_position = 0
        else:
            doc_offset = len(truncated_query) + sequence_added_tokens
            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset


    encoded_dict = tokenizer.encode_plus(
        text=truncated_query,
        pairs=all_doc_tokens,
        truncation='longest_first',
        padding="max_length",
        add_special_tokens=True,
        max_length=max_seq_length
    )

    result = {
        "input_ids": encoded_dict["input_ids"],
        "attention_mask": encoded_dict["attention_mask"],
        "token_type_ids": encoded_dict["token_type_ids"],
        "start_position": start_position,
        "end_position": end_position,
    }
    return result

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


if __name__ == "__main__":
    from cogktr.data.reader.squad2_reader import Squad2Reader

    reader = Squad2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
    train_data, dev_data, _ = reader.read_all()
    vocab = reader.read_vocab()

    processor = Squad2Processor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    print("end")
