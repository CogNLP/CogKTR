from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import AutoTokenizer
from tqdm import tqdm
from cogktr.data.processor.base_processor import BaseProcessor
import transformers
from nltk.corpus import stopwords
import copy

transformers.logging.set_verbosity_error()  # set transformers logging level


class BSemcorProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, addition):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.addition = addition
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.STOPWORDS = list(set(stopwords.words('english')))

    def _remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.STOPWORDS and token.isalnum()]

    def _process(self, data, datatype=None, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")
        for index, item in enumerate(tqdm(self.addition[datatype]["example"])):
            instance_id = item[0]
            instance_defination = item[2].split()
            instance_label = item[3]

            instance_loc = self.addition[datatype]["instance"][instance_id]["instance_loc"]

            sentence_id = instance_id.split(".")[0] + "." + instance_id.split(".")[1]
            raw_sentence = copy.deepcopy(self.addition[datatype]["sentence"][sentence_id]["words"])

            enhanced_data = []


            enhanced_data.extend(instance_defination)
            enhanced_data = self._remove_stopwords(enhanced_data)

            if self.plm == "roberta-large" and self.plm == "roberta-base":
                raise ValueError("roberta will come so on!")
            elif self.plm == "bert-base-cased" or self.plm == "bert-base-uncased":
                input_tokens = []
                raw_sentence.insert(0, '[CLS]')
                raw_sentence.append('[SEP]')
                instance_mask = []
                for index, word in enumerate(raw_sentence):
                    token = self.tokenizer.tokenize(word)
                    input_tokens.extend(token)
                    if index == instance_loc + 1:
                        instance_lens = len(token)
                        instance_mask.extend([1] * len(token))
                    else:
                        instance_mask.extend([0] * len(token))
                sentence_token_len = len(input_tokens)
                enhanced_data.append('[SEP]')
                for word in enhanced_data:
                    token = self.tokenizer.tokenize(word)
                    input_tokens.extend(token)
                sentence_pair_token_len = len(input_tokens) - sentence_token_len
                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
                attention_mask = [1] * len(input_ids)
                segment_ids = [0] * sentence_token_len + [1] * sentence_pair_token_len

                input_ids = input_ids[0:self.max_token_len]
                attention_mask = attention_mask[0:self.max_token_len]
                segment_ids = segment_ids[0:self.max_token_len]
                instance_mask = instance_mask[0:self.max_token_len]

                input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
                attention_mask += [0 for _ in range(self.max_token_len - len(attention_mask))]
                segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
                instance_mask += [0 for _ in range(self.max_token_len - len(instance_mask))]

                datable("input_ids", input_ids)
                datable("attention_mask", attention_mask)
                datable("token_type_ids", segment_ids)
                datable("instance_lens", instance_lens)
                datable("instance_mask", instance_mask)
                datable("label", instance_label)
            else:
                raise ValueError("other plm will come so on!")


        return DataTableSet(datable)

    def process_train(self, data, datatype="train", enhanced_dict=None):
        return self._process(data, datatype=datatype, enhanced_dict=enhanced_dict)

    def process_dev(self, data, datatype="dev", enhanced_dict=None):
        return self._process(data, datatype=datatype, enhanced_dict=enhanced_dict)

    def process_test(self, data, datatype="test", enhanced_dict=None):
        return self._process(data, datatype=datatype, enhanced_dict=enhanced_dict)


if __name__ == "__main__":
    from cogktr.data.reader.temp import TSemcorReader

    reader = TSemcorReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    addition = reader.read_addition()

    processor = BSemcorProcessor(plm="bert-base-cased", max_token_len=512, vocab=vocab, addition=addition)
    dev_dataset = processor.process_dev(dev_data)
    print("end")
