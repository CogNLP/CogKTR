from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import RobertaTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
from cogktr.utils.constant.srl_constant.vocab import TAG_VOCAB
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.constant.conceptnet_constants.constants import *
import numpy as np

transformers.logging.set_verbosity_error()  # set transformers logging level


class OpenBookQAProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, debug=False):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = RobertaTokenizer.from_pretrained(plm)
        self.debug = debug
        self.meta_path_set = set()

    def _process(self, data, enhanced_data_dict=None):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        num_choices = 4
        for i in tqdm(range(len(data)//num_choices)):
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_lis = []
            special_tokens_mask_list = []
            answerKey = data["answerKey"][i * num_choices]
            for j in range(num_choices):
                k = num_choices * i + j
                id,stem,answer_text,key,statement,answerKey = data[k]
                tokenized_data = self.tokenizer.encode_plus(text=stem, text_pair=answer_text,
                                                            truncation='longest_first',
                                                            padding="max_length",
                                                            add_special_tokens=True,
                                                            return_token_type_ids=True,
                                                            return_special_tokens_mask=True,
                                                            max_length=self.max_token_len)
                input_ids = tokenized_data["input_ids"]
                attention_mask = tokenized_data["attention_mask"]
                token_type_ids = tokenized_data["token_type_ids"]
                special_tokens_mask = tokenized_data['special_tokens_mask']

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                special_tokens_mask_list.append(special_tokens_mask)
                token_type_ids_lis.append(token_type_ids)
            # stack the choice features together to create one sample
            datable("input_ids",np.array(input_ids_list,dtype=int))
            datable("attention_mask", np.array(attention_mask_list,dtype=int))
            datable("token_type_ids", np.array(token_type_ids_lis,dtype=int))
            datable("special_tokens_mask",np.array(special_tokens_mask_list,dtype=int))
            datable("label",self.vocab["label_vocab"].label2id(answerKey))
        return DataTableSet(datable)

    def process_train(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_dev(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_test(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)


def encode_meta_path(meta_path_list,meta_path_set,meta_path_vocab):
    total_num_dist_meta_path = len(meta_path_vocab)
    meta_path_feature = np.zeros((total_num_dist_meta_path,ONE_HOT_FEATURE_LENGTH),dtype=np.int8)
    meta_path_count = np.zeros((total_num_dist_meta_path),dtype=np.int8)
    for meta_path in sorted(meta_path_set,key=lambda mp:meta_path_vocab.label2id(mp)):
        idx = meta_path_vocab.label2id(meta_path)
        components = meta_path.split('-')
        start = 0
        for c in components:
            if c in NODE_TYPE_DICT:
                cur = start + NODE_TYPE_DICT[c]
                meta_path_feature[idx][cur] = 1
                start += NODE_TYPE_NUM
            else:
                cur = start + int(c)
                meta_path_feature[idx][cur] = 1
                start += REL_TYPE_NUM
        meta_path_count[idx] = meta_path_list.count(meta_path)
    return meta_path_feature,meta_path_count

def process_roberta(text_a,text_b,tokenizer,max_token_length):
    sep_token = '</s>'
    sequence_a_segment_id = 0
    sequence_b_segment_id = 0
    cls_token = '<s>'
    cls_token_segment_id = 0
    pad_token = 0
    pad_token_segment_id = 0


    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)

    special_tokens_count = 4
    _truncate_seq_pair(tokens_a,tokens_b,max_token_length - special_tokens_count)

    tokens = tokens_a + sep_token + sep_token
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens += tokens_b + sep_token
    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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
        else:
            tokens_b.pop()




