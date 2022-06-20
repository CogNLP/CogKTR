from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
from tqdm import tqdm
import numpy as np
import collections

transformers.logging.set_verbosity_error()  # set transformers logging level


class Squad2SubsetProcessor(BaseProcessor):
    def __init__(self, plm, max_seq_length, max_query_length, doc_stride, vocab, debug=False):
        super().__init__(debug=debug)
        self.plm = plm
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
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

    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _process(self, data, datatype=None):
        datable = DataTable()
        print("Processing data...")
        unique_id = 1000000000
        features = []
        flag = True

        for example_index, (qas_id, question_text,
                            doc_tokens, orig_answer_text,
                            start_position_truth, end_position_truth,
                            is_impossible, que_heads,
                            que_types, que_span,
                            doc_heads, doc_types,
                            all_doc_span, org_doc_token,
                            org_que_token) \
                in tqdm(enumerate(zip(data["qas_id"], data["question_text"],
                                      data["doc_tokens"], data["orig_answer_text"],
                                      data["start_position"], data["end_position"],
                                      data["is_impossible"], data["que_heads"],
                                      data["que_types"], data["que_span"],
                                      data["doc_heads"], data["doc_types"],
                                      data["doc_span"], data["token_doc"],
                                      data["token_que"])), total=len(data["qas_id"])):

            if example_index > 5:
                break
            query_tokens = self.tokenizer.tokenize(question_text)

            que_tokens = []
            prev_is_whitespace = True
            for c in question_text:
                if self._is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        que_tokens.append(c)
                    else:
                        que_tokens[-1] += c
                    prev_is_whitespace = False

            sub_que_span = []

            que_org_to_split_map = {}
            pre_tok_len = 0
            for idx, que_token in enumerate(que_tokens):
                sub_que_tok = self.tokenizer.tokenize(que_token)
                que_org_to_split_map[idx] = (pre_tok_len, len(sub_que_tok) + pre_tok_len - 1)
                pre_tok_len += len(sub_que_tok)

            for idx, (start_ix, end_ix) in enumerate(que_span):
                head_start, head_end = que_org_to_split_map[idx]

                # sub_start_idx and sub_end_idx of children of head node
                head_spans = [(que_org_to_split_map[start_ix - 1][0], que_org_to_split_map[end_ix - 1][1])]
                # all other head sub_tok point to first head sub_tok
                if head_start != head_end:
                    head_spans.append((head_start + 1, head_end))
                    sub_que_span.append(head_spans)

                    for i in range(head_start + 1, head_end + 1):
                        sub_que_span.append([(i, i)])
                else:
                    sub_que_span.append(head_spans)

            assert len(sub_que_span) == len(query_tokens)

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []

            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            doc_org_to_split_map = {}
            pre_tok_len = 0
            for idx, doc_token in enumerate(doc_tokens):
                sub_doc_tok = self.tokenizer.tokenize(doc_token)
                doc_org_to_split_map[idx] = (pre_tok_len, len(sub_doc_tok) + pre_tok_len - 1)
                pre_tok_len += len(sub_doc_tok)

            cnt_span = 0
            for sen_idx, sen_span in enumerate(all_doc_span):
                for idx, (start_ix, end_ix) in enumerate(sen_span):
                    assert (start_ix <= len(sen_span) and end_ix <= len(sen_span))
                    cnt_span += 1

            assert cnt_span == len(doc_tokens)

            sub_doc_span = []
            pre_sen_len = 0
            for sen_idx, sen_span in enumerate(all_doc_span):
                sen_offset = pre_sen_len
                pre_sen_len += len(sen_span)
                for idx, (start_ix, end_ix) in enumerate(sen_span):
                    head_start, head_end = doc_org_to_split_map[sen_offset + idx]
                    # sub_start_idx and sub_end_idx of children of head node
                    head_spans = [(doc_org_to_split_map[sen_offset + start_ix - 1][0],
                                   doc_org_to_split_map[sen_offset + end_ix - 1][1])]
                    # all other head sub_tok point to first head sub_tok
                    if head_start != head_end:
                        head_spans.append((head_start + 1, head_end))
                        sub_doc_span.append(head_spans)

                        for i in range(head_start + 1, head_end + 1):
                            sub_doc_span.append([(i, i)])
                    else:
                        sub_doc_span.append(head_spans)

            assert len(sub_doc_span) == len(all_doc_tokens)

            # making masks
            que_span_mask = np.zeros((len(sub_que_span), len(sub_que_span)))
            for idx, span_list in enumerate(sub_que_span):
                for (start_ix, end_ix) in span_list:
                    if start_ix != end_ix:
                        que_span_mask[start_ix:end_ix + 1, idx] = 1

            doc_span_mask = np.zeros((len(sub_doc_span), len(sub_doc_span)))

            for idx, span_list in enumerate(sub_doc_span):
                for (start_ix, end_ix) in span_list:
                    if start_ix != end_ix:
                        doc_span_mask[start_ix:end_ix + 1, idx] = 1

            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]
                que_span_mask = que_span_mask[:self.max_query_length, :self.max_query_length]

            tok_start_position = None
            tok_end_position = None
            if datatype == "train" and is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if datatype == "train" and not is_impossible:
                tok_start_position = orig_to_tok_index[start_position_truth]
                if end_position_truth < len(doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[end_position_truth + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, self.tokenizer,
                    orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
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
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                # use this idx list to select from doc span mask
                head_select_idx = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                                split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    head_select_idx.append(split_token_index)

                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)
                start_doc_ix = head_select_idx[0]
                end_doc_ix = head_select_idx[-1]
                select_doc_len = end_doc_ix - start_doc_ix + 1
                select_que_len = len(query_tokens)
                assert len(head_select_idx) == select_doc_len

                input_span_mask = np.zeros((self.max_seq_length, self.max_seq_length))
                # 0 count for [CLS] and select_doc_len+1 count for [SEP]
                input_span_mask[1:select_doc_len + 1, 1:select_doc_len + 1] = doc_span_mask[start_doc_ix:end_doc_ix + 1,
                                                                              start_doc_ix:end_doc_ix + 1]
                input_span_mask[select_doc_len + 2:select_doc_len + select_que_len + 2,
                select_doc_len + 2:select_doc_len + select_que_len + 2] = que_span_mask
                record_mask = []
                for i in range(self.max_seq_length):
                    i_mask = []
                    for j in range(self.max_seq_length):
                        if input_span_mask[i, j] == 1:
                            i_mask.append(j)
                    record_mask.append(i_mask)

                for idx, token in enumerate(query_tokens):
                    tokens.append(token)
                    segment_ids.append(1)

                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length
                assert len(input_span_mask) == self.max_seq_length

                start_position = None
                end_position = None
                is_impossible = is_impossible

                if datatype == "train" and not is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if (start_position_truth < doc_start or
                            end_position_truth < doc_start or
                            start_position_truth > doc_end or
                            end_position_truth > doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        is_impossible = True

                    else:
                        # doc_offset = len(query_tokens) + 2
                        doc_offset = 1  # [CLS]
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                        is_impossible = False

                if datatype == "train" and is_impossible:
                    start_position = 0
                    end_position = 0
                    is_impossible = True

                datable("unique_id", unique_id)
                datable("example_index", example_index)
                datable("doc_span_index", doc_span_index)
                datable("tokens", tokens)
                datable("token_to_orig_map", token_to_orig_map)
                datable("token_is_max_context", token_is_max_context)
                datable("input_ids", input_ids)
                datable("input_mask", input_mask)
                datable("segment_ids", segment_ids)
                datable("start_position", start_position)
                datable("end_position", end_position)
                datable("is_impossible", is_impossible)
                datable("input_span_mask", record_mask)
                unique_id += 1

        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data, datatype="train")

    def process_dev(self, data):
        return self._process(data, datatype="dev")

    def process_test(self, data):
        pass


if __name__ == "__main__":
    from cogktr.data.reader.squad2_subset_reader import Squad2SubsetReader

    reader = Squad2SubsetReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0_subset/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = Squad2SubsetProcessor(plm="bert-base-cased", max_seq_length=384, max_query_length=64, doc_stride=128,
                                      vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    print("end")
