from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
from ..qnli_processors import process_sembert
from cogktr.utils.constant.srl_constant.vocab import TAG_VOCAB
from cogktr.utils.vocab_utils import Vocabulary

class Sst5ForSembertProcessor(BaseProcessor):
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
        for sentence, label in tqdm(zip(data['sentence'], data['label']),
                                              total=len(data['sentence'])):
            dict_data =process_sembert(sentence,None,label,self.tokenizer,self.vocab,self.max_token_len,enhanced_data_dict)
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

    def process_test(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence in tqdm(zip(data['sentence']), total=len(data['sentence'])):
            token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
                                          max_length=self.max_token_len)
            datable("token", token)

