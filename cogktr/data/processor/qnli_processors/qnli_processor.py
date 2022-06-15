from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
from cogktr.enhancers.tagger.srl_tagger import SrlTagger,TagTokenizer

transformers.logging.set_verbosity_error()  # set transformers logging level


class QnliProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, debug=False):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)
        self.debug = debug
        self.srl_tagger = SrlTagger(tool="allennlp")
        self.tag_tokenizer = TagTokenizer()

    def debug_process(self, data):
        if self.debug:
            debug_data = DataTable()
            for header in data.headers:
                debug_data[header] = data[header][:100]
            return debug_data
        return data

    def _process(self, data,enhanced_data_dict=None):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        for sentence, question, label in tqdm(zip(data['sentence'], data['question'], data['label']),
                                              total=len(data['sentence'])):
            dict_data = process_sembert(sentence,question,label,self.tokenizer,self.vocab,self.max_token_len,self.srl_tagger,enhanced_data_dict,self.tag_tokenizer)

            datable("input_ids", dict_data["input_ids"])
            datable("input_mask", dict_data["input_mask"])
            datable("token_type_ids", dict_data["token_type_ids"])
            datable("input_tag_ids", dict_data["input_tag_ids"])
            datable("start_end_idx", dict_data["start_end_idx"])
            datable("label", dict_data["label"])

        return DataTableSet(datable)

    def process_train(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_dev(self, data,enhanced_data_dict=None):
        return self._process(data,enhanced_data_dict)

    def process_test(self, data,enhanced_data_dict=None):
        return self._process(data,enhanced_data_dict)


def process_sembert(text_a,text_b,label,tokenizer,vocab,max_token_length,tagger,enhanced_data_dict,tag_tokenizer):

    # text_a_tag_dict = tagger.tag(text_a)
    text_a_tag_dict = enhanced_data_dict[text_a]["srl"]
    tok_to_orig_index_a = [] # tokens -> words
    tokens_a = []
    tokens_a_org = text_a_tag_dict["words"]
    tok_to_orig_index_a.append(0) # [CLS]

    for (i,token) in enumerate(tokens_a_org):
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index_a.append(i+1)
            tokens_a.append(sub_token)

    tok_to_orig_index_b = [] # tokens -> words
    tokens_b = []
    text_b_tag_dict = {}
    if text_b:
        # text_b_tag_dict = tagger.tag(text_b)
        text_b_tag_dict = enhanced_data_dict[text_b]["srl"]
        tokens_b_orig = text_b_tag_dict["words"]
        for (i,token) in enumerate(tokens_b_orig):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index_b.append(i)
                tokens_b.append(sub_token)

        # [CLS] text_A [SEP] text_B [SEP]
        if len(tokens_a + tokens_b) > max_token_length - 3:
            print("Too Long!", len(tokens_a + tokens_b), len(tokens_a), len(tokens_b))

        # 这里是不是应该到减去3？好像还是减去2？
        _truncate_seq_pair(tokens_a,tokens_b,tok_to_orig_index_a,tok_to_orig_index_b,max_token_length - 3)

    else:
        if len(tokens_a) > max_token_length - 2:
            print("Too Long!",len(tokens_a))
            tokens_a = tokens_a[:(max_token_length - 2)]
            tok_to_orig_index_a = tok_to_orig_index_a[:max_token_length - 1]
    tok_to_orig_index_a.append(tok_to_orig_index_a[-1] + 1) # [SEP]
    over_tok_to_orig_index = tok_to_orig_index_a
    if text_b:
        tok_to_orig_index_b.append(tok_to_orig_index_b[-1] + 1) # [SEP]
        offset = tok_to_orig_index_a[-1]
        for org_ix in tok_to_orig_index_b:
            over_tok_to_orig_index.append(offset + org_ix + 1)

    tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
    token_type_ids = [0] * len(tokens)
    len_seq_a = tok_to_orig_index_a[len(tokens) - 1] + 1
    len_seq_b = None
    if text_b:
        tokens += tokens_b + [tokenizer.sep_token]
        len_seq_b = tok_to_orig_index_b[len(tokens_b)] + 1
        token_type_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    pre_ix = -1
    start_split_ix = -1
    over_token_to_orig_map_org = []
    for value in over_tok_to_orig_index:
        over_token_to_orig_map_org.append(value)
    orig_to_token_split_idx = []
    for token_ix, org_ix in enumerate(over_token_to_orig_map_org):
        if org_ix != pre_ix:
            pre_ix = org_ix
            end_split_ix = token_ix - 1
            if start_split_ix != -1:
                orig_to_token_split_idx.append((start_split_ix, end_split_ix))
            start_split_ix = token_ix
    if start_split_ix != -1:
        orig_to_token_split_idx.append((start_split_ix, token_ix))
    while len(orig_to_token_split_idx) < max_token_length:
        orig_to_token_split_idx.append((-1, -1))
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)


    # Zero-pad up to the sequence length.
    padding = [0] * (max_token_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    token_type_ids += padding

    assert len(input_ids) == max_token_length
    assert len(input_mask) == max_token_length
    assert len(token_type_ids) == max_token_length

    label_id = vocab["label_vocab"].label2id(label)

    # construct tag features:
    # print("Start Debug!")


    # aspect padding
    input_tag_ids = []
    sent_tags_list_a = list(text_a_tag_dict["labels"].values())
    if len(sent_tags_list_a) == 0:
        sent_tags_list_a = ["O"] * len(text_a_tag_dict["words"])
    tag_ids_list_a = convert_tags_to_ids(sent_tags_list_a,tag_tokenizer)
    if text_b:
        sent_tags_list_b = list(text_b_tag_dict["labels"].values())
        if len(sent_tags_list_b) == 0:
            sent_tags_list_b = ["O"] * len(text_b_tag_dict["words"])
        tag_ids_list_b = convert_tags_to_ids(sent_tags_list_b,tag_tokenizer)
        input_que_tag_ids = []
        for idx, query_tag_ids in enumerate(tag_ids_list_a):
            query_tag_ids = [1] + query_tag_ids[:len_seq_a - 2] + [2]  # CLS and SEP
            input_que_tag_ids.append(query_tag_ids)
            # construct input doc tag ids with same length as input ids
        for idx, doc_tag_ids in enumerate(tag_ids_list_b):
            tmp_input_tag_ids = input_que_tag_ids[idx]
            doc_input_tag_ids = doc_tag_ids[:len_seq_b - 1] + [2]  # SEP
            input_tag_id = tmp_input_tag_ids + doc_input_tag_ids
            while len(input_tag_id) < max_token_length:
                input_tag_id.append(0)
            assert len(input_tag_id) == len(input_ids)
            input_tag_ids.append(input_tag_id)

    else:
        for idx,query_tag_ids in enumerate(tag_ids_list_a):
            query_tag_ids = [1] + query_tag_ids[:len_seq_a-2] + [2]
            input_tag_id = query_tag_ids
            while len(input_tag_id) < max_token_length:
                input_tag_id.append(0)
            assert len(input_tag_id) == len(input_ids)
            input_tag_ids.append(input_tag_id)

    return {
        "input_ids":input_ids,
        "input_mask":input_mask,
        "token_type_ids":token_type_ids,
        "input_tag_ids":input_tag_ids,
        "start_end_idx":orig_to_token_split_idx,
        "label":label_id,
    }



def _truncate_seq_pair(tokens_a, tokens_b, tok_to_orig_index_a, tok_to_orig_index_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            tok_to_orig_index_a.pop()
        else:
            tokens_b.pop()
            tok_to_orig_index_b.pop()


def convert_tags_to_ids(sent_tags_list,tag_tokenizer,max_num_aspect=3):
    # padding to max_num_aspect
    if len(sent_tags_list) > max_num_aspect:
        sent_tags_list = sent_tags_list[:max_num_aspect]
    tag_set_to_copy = choose_tag_set(sent_tags_list)
    while len(sent_tags_list) < max_num_aspect:
        sent_tags_list.append(tag_set_to_copy)

    sent_tag_ids_list = []
    for sent_tags in sent_tags_list:
        sent_tag_ids = tag_tokenizer.convert_tags_to_ids(sent_tags)
        sent_tag_ids_list.append(sent_tag_ids)
    return sent_tag_ids_list


def choose_tag_set(tag_sets):
    cnt_tag = 0
    tag_ix = 0
    for ix, tag_set in enumerate(tag_sets):
        cnt_tmp_tag = 0
        for tag in tag_set:
            if tag != 'O':
                cnt_tmp_tag = cnt_tmp_tag + 1
        if cnt_tmp_tag > cnt_tag:
            cnt_tag = cnt_tmp_tag
            tag_ix = ix
    chosen_tag_set = tag_sets[tag_ix]

    return chosen_tag_set


if __name__ == "__main__":
    from cogktr.data.reader.qnli_reader import QnliReader

    reader = QnliReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = QnliProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
