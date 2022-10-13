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
import collections
import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


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

    def _process(self, data, isTraining):
        datable = DataTable()
        print("Processing data...")
        data = self.debug_process(data)
        for qas_id, is_impossible, \
            question_text, context_text, \
            answer_text,answers, start_position, end_position, \
            doc_tokens, title \
                in tqdm(zip(data["qas_id"], data["is_impossible"],
                            data["question_text"], data["context_text"],
                            data["answer_text"], data["answers"],
                            data["start_position"],data["end_position"],
                            data["doc_tokens"],data["title"], ),
                        total=len(data["qas_id"])):
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
                "answers":answers,
            }
            example = Namespace(**example)
            results = prepare_mrc_input(example,
                                  tokenizer=self.tokenizer,
                                  max_seq_length=self.max_token_len,
                                  doc_stride=128,
                                  max_query_length=64,
                                  is_training=isTraining)
            for result in results:
                for key,value in result.items():
                    datable(key,value)
                datable("example",example)
            # if len(datable["input_ids"]) != len(datable["example"]):
            #     print("????")
        datable.add_not2torch("example")
        datable.add_not2torch("additional_info")
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data,isTraining=True)

    def process_dev(self, data):
        return self._process(data,isTraining=False)

    def process_test(self, data):
        raise ValueError("Test data is not publicly available!")

    def _collate(self, batch):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return self._collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            try:
                return elem_type({key: self._collate([d[key] for d in batch]) for key in elem})
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {key: self._collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self._collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

            if isinstance(elem, tuple):
                return [self._collate(samples) for samples in transposed]  # Backwards compatibility.
            else:
                try:
                    return elem_type([self._collate(samples) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self._collate(samples) for samples in transposed]

        elif isinstance(elem,Namespace):
            return batch

def prepare_mrc_input(example, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    _DocSpan = collections.namedtuple(
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    results = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if is_training and not example.is_impossible:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        if is_training and example.is_impossible:
            start_position = 0
            end_position = 0
        additional_info = Namespace(**{
            "token_to_orig_map": token_to_orig_map,
            "token_is_max_context": token_is_max_context,
            "tokens": tokens,
        })
        results.append({
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "start_position": start_position,
            "end_position": end_position,
            "additional_info":additional_info,
        })

    return results


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

if __name__ == "__main__":
    from cogktr.data.reader.squad2_reader import Squad2Reader

    reader = Squad2Reader(raw_data_path="/data/hongbang/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
    train_data, dev_data, _ = reader.read_all()
    vocab = reader.read_vocab()

    processor = Squad2Processor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
    # train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    print("end")
