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
from transformers import AutoTokenizer
transformers.logging.set_verbosity_error()  # set transformers logging level


class CommonsenseQAForSafeProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, debug=False):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.debug = debug
        self.meta_path_set = set()

    def _process(self, data, enhanced_data_dict=None):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        num_choices = 5
        for i in tqdm(range(len(data) // num_choices)):
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_lis = []
            meta_path_feature_list = []
            meta_path_count_list = []
            answerKey = data["answerKey"][i * num_choices]
            for j in range(num_choices):
                k = num_choices * i + j
                id, stem, answer_text, key, statement, answerKey = data[k]
                data_dict = enhanced_data_dict[(statement, answer_text)]["interaction"]
                # data_dict = enhanced_data_dict[statement]
                meta_path_feature, meta_path_count = encode_meta_path(
                    data_dict["meta_paths_list"], data_dict["meta_paths_set"], self.vocab["metapath"]
                )

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
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_lis.append(token_type_ids)
                meta_path_feature_list.append(meta_path_feature)
                meta_path_count_list.append(meta_path_count)
            # stack the choice features together to create one sample
            datable("input_ids", np.array(input_ids_list, dtype=int))
            datable("attention_mask", np.array(attention_mask_list, dtype=int))
            datable("token_type_ids", np.array(token_type_ids_lis, dtype=int))
            datable("meta_path_feature", np.array(meta_path_feature_list, dtype=np.float32))
            datable("meta_path_count", np.array(meta_path_count_list, dtype=np.float32))
            meta_path_vocab = self.vocab["label_vocab"]
            datable("label", meta_path_vocab.label2id(answerKey))

        return DataTableSet(datable)

    def process_train(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_dev(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_test(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)


def encode_meta_path(meta_path_list, meta_path_set, meta_path_vocab):
    total_num_dist_meta_path = len(meta_path_vocab)
    meta_path_feature = np.zeros((total_num_dist_meta_path, ONE_HOT_FEATURE_LENGTH), dtype=np.int8)
    meta_path_count = np.zeros((total_num_dist_meta_path), dtype=np.int8)
    for meta_path in sorted(meta_path_set, key=lambda mp: meta_path_vocab.label2id(mp)):
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
    return meta_path_feature, meta_path_count

