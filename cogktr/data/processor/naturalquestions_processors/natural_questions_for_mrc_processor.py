from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
from argparse import Namespace
from cogktr.data.processor.squad2_processors.squad2_processor import prepare_mrc_input
transformers.logging.set_verbosity_error()  # set transformers logging level

class NaturalQuestionsForMRCProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data,enhanced_data_dict):
        datable = DataTable()
        print("Processing data...")
        for question_id,(question, answers) in tqdm(enumerate(zip(data['question'], data['answers'])), total=len(data['question'])):
            for psg_id,psg in enumerate(enhanced_data_dict[question]["passages"]):
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                context_text = psg[1]

                # Split on whitespace so that different tokens may be attributed to their original position.
                for c in context_text:
                    if _is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)
                example = Namespace(**{"question_text":question,
                                       "doc_tokens":doc_tokens,
                                       "qas_id":str(question_id)+'-'+str(psg_id),
                                       "answers":{"text":answer for answer in answers},
                                       "is_impossible":False,
                                       })
                results = prepare_mrc_input(example,
                                            tokenizer=self.tokenizer,
                                            max_seq_length=self.max_token_len,
                                            doc_stride=128,
                                            max_query_length=64,
                                            is_training=False)
                for result in results:
                    for key, value in result.items():
                        datable(key, value)
                    datable("example", example)
        datable.add_not2torch("example")
        datable.add_not2torch("additional_info")
        return DataTableSet(datable)

    def process_train(self, data,enhanced_data_dict):
        return self._process(data,enhanced_data_dict)

    def process_dev(self, data,enhanced_data_dict):
        return self._process(data,enhanced_data_dict)

    def process_test(self,data,enhanced_data_dict):
        return self._process(data,enhanced_data_dict)

    # def process_test(self, data):
    #     datable = DataTable()
    #     print("Processing data...")
    #     for sentence in tqdm(zip(data['sentence']), total=len(data['sentence'])):
    #         token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
    #                                       max_length=self.max_token_len)
    #         tokenized_data = self.tokenizer.encode_plus(text=sentence,
    #                                                     padding="max_length",
    #                                                     add_special_tokens=True,
    #                                                     max_length=self.max_token_len)
    #         datable("input_ids", tokenized_data["input_ids"])
    #         datable("token_type_ids", tokenized_data["token_type_ids"])
    #         datable("attention_mask", tokenized_data["attention_mask"])
    #     return DataTableSet(datable)

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


if __name__ == "__main__":
    from cogktr import *
    from cogktr.data.reader.sst2_reader import Sst2Reader
    from cogktr.data.reader.naturalquestion_reader import NaturalQuestionsReader

    reader = NaturalQuestionsReader(
        raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data",
        debug=True
    )
    train_data, dev_data, test_data = reader.read_all()

    enhancer = WorldEnhancer(
        knowledge_graph_path="/data/hongbang/CogKTR/datapath/knowledge_graph",
        cache_path="/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/enhanced_data",
        cache_file="world_data",
        reprocess=False,
        load_entity_desc=False,
        load_entity_embedding=False,
        load_entity_kg=False,
        load_retrieval_info=True
    )

    top_k_relevant_psgs = 10
    enhanced_train_dict = enhancer.enhance_train(train_data, enhanced_key_1='question',
                                                 return_top_k_relevent_psgs=top_k_relevant_psgs)
    enhanced_dev_dict = enhancer.enhance_dev(dev_data, enhanced_key_1='question',
                                             return_top_k_relevent_psgs=top_k_relevant_psgs)
    enhanced_test_dict = enhancer.enhance_test(test_data, enhanced_key_1='question',
                                               return_top_k_relevent_psgs=top_k_relevant_psgs)

    processor = NaturalQuestionsForMRCProcessor(
        plm="bert-base-uncased",
        max_token_len=100,
    )
    train_dataset = processor.process_train(train_data,enhanced_train_dict)
    dev_dataset = processor.process_dev(dev_data,enhanced_dev_dict)
    test_dataset = processor.process_test(test_data, enhanced_test_dict)
