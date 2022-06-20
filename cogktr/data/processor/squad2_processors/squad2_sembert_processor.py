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
from ..qnli_processors.qnli_processor import process_sembert
from cogktr.enhancers.tagger.srl_tagger import SrlTagger,TagTokenizer

transformers.logging.set_verbosity_error()  # set transformers logging level


class Squad2SembertProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, debug=False):
        super().__init__(debug=debug)
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)
        self.tag_tokenizer = TagTokenizer()
        self.vocab["tag_vocab"] = self.tag_tokenizer.tag_vocab

    def _process(self, data,enhanced_data_dict=None):
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
            dict_data = process_sembert(question_text, context_text, None, self.tokenizer, self.vocab, self.max_token_len,
                            enhanced_data_dict,self.tag_tokenizer)
            datable("input_ids", dict_data["input_ids"])
            datable("input_mask", dict_data["input_mask"])
            datable("token_type_ids", dict_data["token_type_ids"])
            datable("input_tag_ids", dict_data["input_tag_ids"])
            datable("start_end_idx", dict_data["start_end_idx"])
            datable("start_position", dict_data["start_position"])
            datable("end_position", dict_data["end_position"])
        return DataTableSet(datable)

    def process_train(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_dev(self, data,enhanced_data_dict=None):
        return self._process(data,enhanced_data_dict)


    def process_test(self, data):
        return self._process(data)



