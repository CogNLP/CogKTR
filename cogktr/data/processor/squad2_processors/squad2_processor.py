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


transformers.logging.set_verbosity_error()  # set transformers logging level

class Squad2Processor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def process(self, data):
        datable = DataTable()
        print("Processing data...")
        dict_data = {}
        for head in data.headers:
            dict_data[head] = data[head][0:100]
        data = dict_data
        for qas_id,is_impossible, \
            question_text,context_text,\
            answer_text,start_position,end_position,\
            doc_tokens,title \
            in tqdm(zip(data["qas_id"],data["is_impossible"],
                   data["question_text"],data["context_text"],
                   data["answer_text"],data["start_position"],
                   data["end_position"],data["doc_tokens"],
                   data["title"],),total=len(data["qas_id"])):
            example = {
                "qas_id":qas_id,
                "is_impossible":is_impossible,
                "question_text":question_text,
                "context_text":context_text,
                "answer_text":answer_text,
                "start_position":start_position,
                "end_position":end_position,
                "doc_tokens":doc_tokens,
                "title":title,
            }
            example = Namespace(**example)
            result = process_bert(example,
                         tokenizer=self.tokenizer,
                         max_seq_length=self.max_token_len,
                         doc_stride=128,
                         max_query_length=64,
                         padding_strategy="max_length",
                         is_training=True)
            for key,value in result.items():
                datable(key,value)

        return DataTableSet(datable)

    def process_train(self, data):
        return self.process(data)

    def process_dev(self, data):
        return self.process(data)

    def process_test(self, data):
        return self.process(data)

def process_bert(example,tokenizer,max_seq_length, doc_stride, max_query_length, padding_strategy, is_training):
    start_position = example.start_position
    end_position = example.end_position

    encoded_dict = tokenizer.encode_plus(
        text = example.context_text,
        pairs = example.answer_text,
        truncation='longest_first',
        padding="max_length",
        add_special_tokens=True,
        max_length=max_seq_length
    )
    result = {
        "input_ids": encoded_dict["input_ids"],
        "attention_mask": encoded_dict["attention_mask"],
        "token_type_ids": encoded_dict["token_type_ids"],
        "start_position": start_position,
        "end_position": end_position,
    }
    return result

if __name__ == "__main__":
    from cogktr.data.reader.squad2_reader import Squad2Reader

    reader = Squad2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
    train_data, dev_data, _ = reader.read_all()
    vocab = reader.read_vocab()
    processor = Squad2Processor(plm="bert-base-cased", max_token_len=256, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    print("end")