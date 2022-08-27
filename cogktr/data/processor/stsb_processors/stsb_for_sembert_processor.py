from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
from cogktr.data.processor.base_processor import BaseProcessor
import transformers
from ..qnli_processors.qnli_sembert_processor import process_sembert
from cogktr.utils.constant.srl_constant.vocab import TAG_VOCAB
from cogktr.utils.vocab_utils import Vocabulary

transformers.logging.set_verbosity_error()  # set transformers logging level


class StsbForSembertProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab,debug=False):
        super().__init__(debug)
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)
        tag_vocab = Vocabulary()
        tag_vocab.add_sequence(TAG_VOCAB)
        tag_vocab.create()
        self.vocab["tag_vocab"] = tag_vocab

    def _process(self, data,enhanced_data_dict=None):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        for sentence1, sentence2, score in tqdm(zip(data['sentence1'], data['sentence2'], data['score']),
                                                total=len(data['score'])):
            dict_data = process_sembert(sentence1,sentence2,score,self.tokenizer,self.vocab,self.max_token_len,enhanced_data_dict)
            datable("input_ids", dict_data["input_ids"])
            datable("input_mask", dict_data["input_mask"])
            datable("token_type_ids", dict_data["token_type_ids"])
            datable("input_tag_ids", dict_data["input_tag_ids"])
            datable("start_end_idx", dict_data["start_end_idx"])
            datable("label", float(score))
        return DataTableSet(datable)

    def process_train(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_dev(self, data,enhanced_data_dict=None):
        return self._process(data,enhanced_data_dict)

    def process_test(self, data, enhanced_data_dict=None):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        for sentence1, sentence2, in tqdm(zip(data['sentence1'], data['sentence2']),
                                                total=len(data['sentence1'])):
            dict_data = process_sembert(sentence1,sentence2,None,self.tokenizer,self.vocab,self.max_token_len,enhanced_data_dict)
            datable("input_ids", dict_data["input_ids"])
            datable("input_mask", dict_data["input_mask"])
            datable("token_type_ids", dict_data["token_type_ids"])
            datable("input_tag_ids", dict_data["input_tag_ids"])
            datable("start_end_idx", dict_data["start_end_idx"])
        return DataTableSet(datable)



