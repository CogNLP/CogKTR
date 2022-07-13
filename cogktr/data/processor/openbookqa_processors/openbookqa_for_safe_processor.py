from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
from cogktr.utils.constant.srl_constant.vocab import TAG_VOCAB
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.constant.conceptnet_constants.constants import *
import numpy as np

transformers.logging.set_verbosity_error()  # set transformers logging level


class OpenBookQAForSafeProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, debug=False):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)
        self.debug = debug
        self.meta_path_set = set()

    def _process(self, data, enhanced_data_dict=None):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        for i in tqdm(range(len(data["statement"]))):
            id,stem,answer_text,key,statement,answerKey = data[i]
            data_dict = enhanced_data_dict[statement]
            meta_path_feature,meta_path_count = encode_meta_path(
                data_dict["meta_paths_list"],data_dict["meta_paths_set"],self.vocab["metapath"]
            )

            # datable("input_ids", dict_data["input_ids"])
            # datable("input_mask", dict_data["input_mask"])
            # datable("token_type_ids", dict_data["token_type_ids"])
            # datable("input_tag_ids", dict_data["input_tag_ids"])
            # datable("start_end_idx", dict_data["start_end_idx"])
            # datable("label", dict_data["label"])

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





