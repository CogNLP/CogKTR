from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from tqdm import tqdm
from transformers import BertTokenizer
import transformers
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class Conll2005SrlSubsetProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, max_label_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.max_label_len = max_label_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence_number, words, pos_tags, verb_indicator, dep_head, dep_label, tags, metadata in tqdm(
                zip(data['sentence_id'], data['tokens'], data['pos_tags'], data['verb_indicator'],
                    data['dep_head'], data['dep_label'], data['tags'], data['metadata']),
                total=len(data['sentence_id'])):

            dep_rel = [self.vocab["dep_label_vocab"].label2id(t.lower()) for t in dep_label]
            dep_head = [int(x) for x in dep_head]
            labels = tags[0]
            verb_index = metadata['verb_index']

            tokens = []
            subword_token_len = []
            word_indexs = []

            index = 0
            for word in words:
                token = self.tokenizer.tokenize(word)
                tokens += token
                subword_token_len.append(len(token))
                word_indexs.append(index)
                index += len(token)

            alignment = []
            for i, l in zip(word_indexs, subword_token_len):
                assert l > 0
                aligned_subwords = []
                for j in range(l):
                    aligned_subwords.append(i + j)
                alignment.append(aligned_subwords)

            if verb_index:
                verb_indicator = [0] * (sum(map(lambda x: len(x), alignment[:verb_index])) + 1) + \
                                 [1] * len(alignment[verb_index]) + \
                                 [0] * (sum(map(lambda x: len(x), alignment[verb_index + 1:])) + 1)

            tokens = tokens + ['[SEP]']
            segment_ids = [0] * len(tokens)
            subword_token_len = subword_token_len + [1]
            word_indexs.append(len(tokens) - 1)

            tokens = ['[CLS]'] + tokens
            segment_ids = [0] + segment_ids
            subword_token_len = [1] + subword_token_len
            word_indexs = [0] + [i + 1 for i in word_indexs]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            alignment = [[val + 1 for val in list_] for list_ in alignment]

            align_sizes = [0 for _ in range(len(tokens))]
            wp_rows = []
            for word_piece_slice in alignment:
                wp_rows.append(word_piece_slice)
                for i in word_piece_slice:
                    align_sizes[i] += 1
            offset = 0
            for i in range(len(tokens)):
                if align_sizes[offset + i] == 0:
                    align_sizes[offset + i] = len(words)
                    for j in range(len(words)):
                        wp_rows[j].append(offset + i)

            wp_dep_head, wp_dep_rel = [], []
            for i, (idx, slen) in enumerate(zip(word_indexs, subword_token_len)):
                if i == 0 or i == len(subword_token_len) - 1:
                    wp_dep_rel.append(self.vocab["dep_label_vocab"].label2id('special_rel'))
                    wp_dep_head.append(idx + 1)
                else:
                    rel = dep_rel[i - 1]
                    wp_dep_rel.append(rel)
                    head = dep_head[i - 1]
                    if head == 0:
                        wp_dep_head.append(0)
                    else:
                        if head < max(word_indexs):
                            new_pos = word_indexs[head - 1 + 1]
                        else:
                            new_pos = idx + 1
                        wp_dep_head.append(new_pos + 1)

                    for _ in range(1, slen):
                        wp_dep_rel.append(self.vocab["dep_label_vocab"].label2id('subtokens'))
                        wp_dep_head.append(idx + 1)

            input_mask = [1] * len(input_ids)
            padding_length = self.max_token_len - len(input_ids)
            input_ids = input_ids + ([0] * padding_length)
            input_mask = input_mask + ([0] * padding_length)

            if verb_index:
                segment_ids = verb_indicator
            segment_ids = segment_ids + ([0] * padding_length)
            verb_index = verb_index if verb_index else None

            datable("sentence_number", sentence_number)
            datable("words", words)
            datable("input_ids", input_ids)
            datable("input_masks", input_mask)
            datable("segment_ids", segment_ids)
            datable("dep_rel", dep_rel)
            datable("dep_head", dep_head)
            datable("wp_rows", wp_rows)
            datable("align_sizes", align_sizes)
            datable("tokens_len", len(tokens))
            datable("labels", labels)
            datable("verb_index", verb_index)

        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)


if __name__ == "__main__":
    from cogktr.data.reader.conll2005_srl_subset_reader import Conll2005SrlSubsetReader

    reader = Conll2005SrlSubsetReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2005_srl_subset/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = Conll2005SrlSubsetProcessor(plm="bert-base-cased", max_token_len=512, max_label_len=512, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
