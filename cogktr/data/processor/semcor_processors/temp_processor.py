from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import AutoTokenizer
from tqdm import tqdm
from cogktr.data.processor.base_processor import BaseProcessor
import transformers
from nltk.corpus import stopwords
import copy

transformers.logging.set_verbosity_error()  # set transformers logging level


class TSemcorProcessor(BaseProcessor):
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
        wordnet_info_list = []
        print("Processing data...")
        for sentence, tag_list, lemma_list, all_pos_list, sentence_id, pos_list, instance_list, instance_id_list in tqdm(
                zip(data['sentence'], data['tag_list'], data['lemma_list'], data['pos_list'],
                    data['sentence_id'], data["instance_pos_list"], data["instance_list"], data["instance_id_list"]),
                total=len(data['sentence'])):
            for index, instance_id in enumerate(instance_id_list):
                instance_wordnet_dict = enhanced_dict[tuple(instance_list)]["wordnet"][index]["lemma_item_details"]
                for instance_candidate, content in instance_wordnet_dict.items():
                    wordnet_info_list.append(content)

        for index, item in enumerate(tqdm(self.addition[datatype]["example"])):
            instance_id = item[0]
            instance_label = item[3]

            instance_loc = self.addition[datatype]["instance"][instance_id]["instance_loc"]

            sentence_id = instance_id.split(".")[0] + "." + instance_id.split(".")[1]
            raw_sentence = copy.deepcopy(self.addition[datatype]["sentence"][sentence_id])

            wordnet_info = wordnet_info_list[index]
            enhanced_data = []
            # synonym = wordnet_info["synonym"]
            # examples = []
            # if len(wordnet_info["examples"]) > 0:
            #     examples = wordnet_info["examples"][0].split()
            definition = wordnet_info["definition"].split()
            # hypernym_examples = []
            # hypernym_definition = []
            # if wordnet_info["hypernym"]["definition"] is not None:
            #     hypernym_examples = wordnet_info["hypernym"]["definition"].split()
            #     if len(wordnet_info["hypernym"]["examples"]) > 0:
            #         hypernym_definition = wordnet_info["hypernym"]["examples"][0].split()
            # enhanced_data.extend(synonym)
            # enhanced_data.extend(examples)
            enhanced_data.extend(definition)
            # enhanced_data.extend(hypernym_examples)
            # enhanced_data.extend(hypernym_definition)
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

            # tokenized_data = self.tokenizer.encode_plus(text=" ".join(raw_sentence),
            #                                             text_pair=" ".join(enhanced_data),
            #                                             truncation='longest_first',
            #                                             padding="max_length",
            #                                             add_special_tokens=True,
            #                                             max_length=self.max_token_len)
            # datable("input_ids", tokenized_data["input_ids"])
            # datable("attention_mask", tokenized_data["attention_mask"])
            # if self.plm != "roberta-large" and self.plm != "roberta-base":
            #     datable("token_type_ids", tokenized_data["token_type_ids"])
            # datable("label", instance_label)
        return DataTableSet(datable)

    def process_train(self, data, datatype="train", enhanced_dict=None):
        return self._process(data, datatype=datatype, enhanced_dict=enhanced_dict)

    def process_dev(self, data, datatype="dev", enhanced_dict=None):
        return self._process(data, datatype=datatype, enhanced_dict=enhanced_dict)

    def process_test(self, data, datatype="test", enhanced_dict=None):
        return self._process(data, datatype=datatype, enhanced_dict=enhanced_dict)


if __name__ == "__main__":
    from cogktr.data.reader.temp import TSemcorReader
    from cogktr.enhancers.linguistics_enhancer import LinguisticsEnhancer

    reader = TSemcorReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    addition = reader.read_addition()

    enhancer = LinguisticsEnhancer(load_wordnet=True,
                                   cache_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/enhanced_data",
                                   cache_file="linguistics_data",
                                   reprocess=True)
    # enhanced_train_dict = enhancer.enhance_train(datable=train_data,
    #                                              return_wordnet=True,
    #                                              enhanced_key_1="instance_list",
    #                                              pos_key="instance_pos_list")
    enhanced_dev_dict = enhancer.enhance_dev(datable=dev_data,
                                             return_wordnet=True,
                                             enhanced_key_1="instance_list",
                                             pos_key="instance_pos_list")

    processor = TSemcorProcessor(plm="bert-base-cased", max_token_len=512, vocab=vocab, addition=addition)
    # train_dataset = processor.process_train(train_data, enhanced_dict=enhanced_train_dict)
    dev_dataset = processor.process_dev(dev_data, enhanced_dict=enhanced_dev_dict)
    print("end")
