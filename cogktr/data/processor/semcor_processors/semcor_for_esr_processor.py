from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import AutoTokenizer
from tqdm import tqdm
from cogktr.data.processor.base_processor import BaseProcessor
import transformers
from nltk.corpus import stopwords
import copy

transformers.logging.set_verbosity_error()  # set transformers logging level


class SemcorForEsrProcessor(BaseProcessor):
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
            instance_key = item[1]
            instance_label = item[3]
            instance_list_item = enhanced_dict[tuple(item[4])]
            item_index = int(item[0].split(".")[2][1:])
            current_instance_item_list = instance_list_item["wordnet"][item_index]['lemma_item_details']
            current_knowledge = None
            for temp_key, temp_value in current_instance_item_list.items():
                if instance_key == temp_value["lemma_key"]:
                    current_knowledge = copy.deepcopy(temp_value)

            instance_loc = self.addition[datatype]["instance"][instance_id]["instance_loc"]

            sentence_id = instance_id.split(".")[0] + "." + instance_id.split(".")[1]
            raw_sentence = copy.deepcopy(self.addition[datatype]["sentence"][sentence_id]["words"])

            enhanced_data = []
            if current_knowledge is not None:
                enhanced_data.extend(current_knowledge["synonym"])
                enhanced_data.extend(current_knowledge["definition"].split())
                for example in (current_knowledge["examples"]):
                    enhanced_data.extend(example.split())
                if current_knowledge['hypernym']["synset"] is not None:
                    enhanced_data.extend(current_knowledge['hypernym']["definition"].split())
                    for example in (current_knowledge['hypernym']["examples"]):
                        enhanced_data.extend(example.split())

                enhanced_data = self._remove_stopwords(enhanced_data)

            if self.plm == "roberta-large" and self.plm == "roberta-base":
                raise ValueError("roberta will come so on!")
            elif self.plm == "bert-base-cased" or self.plm == "bert-base-uncased":
                input_tokens = []
                instance_mask = []
                raw_sentence.insert(0, '[CLS]')
                raw_sentence.append('[SEP]')
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
    from cogktr.data.reader.semcor_reader import SemcorReader
    from cogktr.enhancers.linguistics_enhancer import LinguisticsEnhancer

    reader = SemcorReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    addition = reader.read_addition()

    enhancer = LinguisticsEnhancer(load_wordnet=True,
                                   cache_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/enhanced_data",
                                   cache_file="linguistics_data",
                                   reprocess=False)
    # enhanced_train_dict = enhancer.enhance_train(datable=train_data,
    #                                              return_wordnet=True,
    #                                              enhanced_key_1="instance_list",
    #                                              pos_key="instance_pos_list")
    enhanced_dev_dict = enhancer.enhance_dev(datable=dev_data,
                                             return_wordnet=True,
                                             enhanced_key_1="instance_list",
                                             pos_key="instance_pos_list")

    processor = SemcorForEsrProcessor(plm="bert-base-cased", max_token_len=512, vocab=vocab, addition=addition)
    # train_dataset = processor.process_train(train_data, enhanced_dict=enhanced_train_dict)
    dev_dataset = processor.process_dev(dev_data, enhanced_dict=enhanced_dev_dict)
    print("end")
