import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import xml.etree.ElementTree as ET
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import transformers
from nltk.tokenize import word_tokenize
from collections import Counter, OrderedDict

TAG = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV}
STOPWORDS = list(set(stopwords.words('english')))


class SemcorReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_semcor_raw_data(raw_data_path)
        # self.train_file = 'Training_Corpora/SemCor/semcor.data.xml'
        self.train_file = 'Evaluation_Datasets/semeval2007/semeval2007.data.xml'
        self.train_gold_file = 'Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
        self.dev_file = 'Evaluation_Datasets/semeval2007/semeval2007.data.xml'
        self.dev_gold_file = 'Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
        self.test_file = 'test.tsv'
        self.test_gold_file = 'test.tsv'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.train_gold_path = os.path.join(raw_data_path, self.train_gold_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.dev_gold_path = os.path.join(raw_data_path, self.dev_gold_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.test_gold_path = os.path.join(raw_data_path, self.test_gold_file)
        self.label_vocab = Vocabulary()
        self.segment_list=[]

    def tokenize(self, text):
        return self.tokenizer.encode(' ' + text.strip(), add_special_tokens=False)

    def get_synonym_tokens(self, synset):
        synonym_tokens = []
        for lemma in synset.lemmas():
            synonym_tokens += lemma.name().replace('_', ' ').split()
        return synonym_tokens

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in STOPWORDS and token.isalnum()]

    def get_example_tokens(self, synset):
        example_tokens = []
        for example in synset.examples():
            example_tokens += self.remove_stopwords(word_tokenize(example.lower()))
        return example_tokens

    def get_hypernym_tokens(self, synset):
        hypernym_tokens = []
        for hypernym in synset.hypernyms():
            hypernym_tokens += self.remove_stopwords(word_tokenize(hypernym.definition().lower()))
        return hypernym_tokens

    def get_related_words(self, synset):
        synonym_tokens = self.get_synonym_tokens(synset)
        example_tokens = self.get_example_tokens(synset)
        hypernym_tokens = self.get_hypernym_tokens(synset)
        all_tokens = synonym_tokens + example_tokens + hypernym_tokens
        sorting_counts = Counter(all_tokens)
        all_tokens = sorted(list(OrderedDict.fromkeys(all_tokens)), key=lambda x: sorting_counts[x], reverse=True)
        return ' '.join(all_tokens)

    def get_definition(self, synset):
        return synset.definition() + ' ' + self.get_related_words(synset)

    def get_sentence_id(self, instance_id, offset):
        sentence_id = instance_id.rsplit('.', 1)[0]
        if offset == 0:
            return sentence_id
        else:
            t_id, s_id = sentence_id.rsplit('.', 1)
            length = len(s_id) - 1
            index = int(s_id[1:]) + offset
            if index < 0: return None
            if len(str(index)) > length: return None
            s_id = 's{0:0={1}d}'.format(index, length)
            return '.'.join([t_id, s_id])

    def get_example_list(self):
        return self.example_list

    def get_segment_list(self):
        return self.segment_list

    def probe(self, feature):
        instance_begin, instance_end, _ = feature.instance

    def analyze_tokenizer(self, tokenizer):
        tokenize_stype_dict = {
            'BertTokenizer': True,
            'AlbertTokenizer': True,
            'RobertaTokenizerFast': False,
            'XLMRobertaTokenizer': False,
            'BartTokenizer': False
        }
        self.tokenize_bert_stype = tokenize_stype_dict[type(tokenizer).__name__]
        self.tokenizer = tokenizer

    def _read_data(self, path):
        print("Reading data...")
        tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
        limit = 432
        annotators = [0]
        wsd_label = self.train_gold_path
        datable = DataTable()
        self.analyze_tokenizer(tokenizer)
        begin_id = tokenizer.convert_tokens_to_ids(
            tokenizer.cls_token if self.tokenize_bert_stype else tokenizer.bos_token)
        end_id = tokenizer.convert_tokens_to_ids(
            tokenizer.sep_token if self.tokenize_bert_stype else tokenizer.eos_token)
        col_id = self.tokenize(':')
        col_len = len(col_id)
        root = ET.parse(self.train_path).getroot()
        sentence_dict = {}
        instance_dict = {}
        sentences = root.findall('.//sentence')
        for sentence in tqdm(sentences):
            word_token = []
            word_token_len = []
            index = 0
            for word in sentence.getchildren():
                tokens = self.tokenize(word.text)
                word_token.append(tokens)
                word_token_len.append(len(tokens))
                if word.tag == 'instance':
                    instance_dict[word.get('id')] = index
                index += 1
            sentence_dict[sentence.get('id')] = (word_token, word_token_len)
        if wsd_label is not None:
            label_dict = {}
            with open(wsd_label, 'r') as file:
                for line in file.readlines():
                    line = line.split()
                    label_dict[line[0]] = line[1:]
            self.segment_list = [0]
        self.example_list = []
        gloss_dict = {}
        instances = root.findall('.//instance')
        for instance in tqdm(instances):
            instance_id = instance.get('id')
            lemmas = wn.lemmas(instance.get('lemma'), TAG[instance.get('pos')])
            if wsd_label is None:
                for lemma in lemmas:
                    lemma_key = lemma.key()
                    if not lemma_key in gloss_dict:
                        gloss_dict[lemma_key] = self.tokenize(self.get_definition(lemma.synset()))
                    self.example_list.append((instance_id, lemma_key, None))
            else:
                gold_lemma_keys = label_dict[instance_id]
                if len(gold_lemma_keys) > 1:
                    gold_lemma_keys = [gold_lemma_keys[annotator] for annotator in annotators]
                for gold_lemma_key in gold_lemma_keys:
                    self.example_list.append((instance_id, gold_lemma_key, 1))
                for lemma in lemmas:
                    lemma_key = lemma.key()
                    if not lemma_key in gloss_dict:
                        gloss_dict[lemma_key] = self.tokenize(self.get_definition(lemma.synset()))
                    if lemma_key in gold_lemma_keys: continue
                    self.example_list.append((instance_id, lemma_key, 0))
                self.segment_list.append(len(self.example_list))
        self.features = []
        for instance_id, lemma_key, label in tqdm(self.example_list):
            input_ids = [0] * limit
            if self.tokenize_bert_stype:
                total = col_len + 3
            else:
                total = col_len + 4
            prev_sentence_id = self.get_sentence_id(instance_id, -1)
            if prev_sentence_id in sentence_dict:
                prev_word_token, prev_word_token_len = sentence_dict[prev_sentence_id]
                prev_word_token_len_sum = sum(prev_word_token_len)
                total += prev_word_token_len_sum
            next_sentence_id = self.get_sentence_id(instance_id, 1)
            if next_sentence_id in sentence_dict:
                next_word_token, next_word_token_len = sentence_dict[next_sentence_id]
                next_word_token_len_sum = sum(next_word_token_len)
                total += next_word_token_len_sum
            cur_sentence_id = self.get_sentence_id(instance_id, 0)
            cur_word_token, cur_word_token_len = sentence_dict[cur_sentence_id]
            cur_word_token_len_sum = sum(cur_word_token_len)
            total += cur_word_token_len_sum
            index = instance_dict[instance_id]
            instance_len = cur_word_token_len[index]
            total += instance_len
            gloss_token = gloss_dict[lemma_key]
            gloss_token_len = len(gloss_token)
            total += gloss_token_len
            # [CLS] / <s>
            offset = 0
            input_ids[offset] = begin_id
            offset += 1
            # Previous sentence
            if total <= limit and prev_sentence_id in sentence_dict:
                offset += prev_word_token_len_sum
                input_ids[1:offset] = [token for tokens in prev_word_token for token in tokens]
            # Current sentence
            instance_begin = offset + sum(cur_word_token_len[:index])
            instance_end = instance_begin + instance_len
            input_ids[offset:offset + cur_word_token_len_sum] = [token for tokens in cur_word_token for token in tokens]
            offset += cur_word_token_len_sum
            # Next sentence
            if total <= limit and next_sentence_id in sentence_dict:
                input_ids[offset:offset + next_word_token_len_sum] = [token for tokens in next_word_token for token in
                                                                      tokens]
                offset += next_word_token_len_sum
            # [SEP] / </s> + <s>
            input_ids[offset] = end_id
            offset += 1
            context_len = offset
            if not self.tokenize_bert_stype:
                input_ids[offset] = begin_id
                offset += 1
            # Ambiguous word
            input_ids[offset:offset + instance_len] = cur_word_token[index]
            offset += instance_len
            # Colon
            input_ids[offset:offset + col_len] = col_id
            offset += col_len
            # Gloss
            if offset + gloss_token_len > limit - 1:
                gloss_token_len = limit - 1 - offset
            input_ids[offset:offset + gloss_token_len] = gloss_token[:gloss_token_len]
            offset += gloss_token_len
            # [SEP] / </s>
            input_ids[offset] = end_id
            offset += 1
            assert offset <= limit
            self.label_vocab.add(label)
            datable("input_ids", input_ids[:offset])
            datable("input_len", offset)
            datable("context_len", context_len)
            datable("instance", (instance_begin, instance_end, instance_len))
            datable("label", label)
        return datable

    def _read_train(self, path):
        return self._read_data(path)

    def _read_dev(self, path):
        return self._read_data(path)

    def _read_test(self, path):
        return self._read_data(path)

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = SemcorReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/word_sense_disambiguation/SemCor/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
