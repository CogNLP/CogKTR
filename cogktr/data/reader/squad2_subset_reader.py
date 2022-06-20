import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.io_utils import load_json
from cogktr.utils.download_utils import Downloader
import json
from tqdm import tqdm
import spacy


class SimpleNlp(object):
    def __init__(self):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def nlp(self, text):
        return self.nlp(text)


class Squad2SubsetReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_squad2_raw_data(raw_data_path)
        self.train_file = 'squad_sample.json'
        self.dev_file = 'squad_sample.json'
        self.test_file = ''
        self.train_tag_file = 'squad_span_sample.json'
        self.dev_tag_file = 'squad_span_sample.json'
        self.test_tag_file = ''
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.train_tag_path = os.path.join(raw_data_path, self.train_tag_file)
        self.dev_tag_path = os.path.join(raw_data_path, self.dev_tag_file)
        self.test_tag_path = os.path.join(raw_data_path, self.test_tag_file)

    def _read_data(self, path, tag_path=None, datatype=None):
        datable = DataTable()

        raw_data_dict = load_json(path)
        input_data = raw_data_dict["data"]

        input_tag_data = []
        with open(tag_path, "r", encoding='utf-8') as file:
            for line in file:
                input_tag_data.append(json.loads(line))

        qas_id_to_tag_idx_map = {}
        all_dqtag_data = []
        for idx, tag_data in enumerate(tqdm(input_tag_data, ncols=50)):
            qas_id = tag_data["qas_id"]
            qas_id_to_tag_idx_map[qas_id] = idx
            tag_rep = tag_data["tag_rep"]
            dqtag_data = {
                "qas_id": qas_id,
                "head_que": [int(i) for i in tag_rep["pred_head_que"]],
                "span_que": [eval(i) for i in tag_rep["hpsg_list_que"]],
                "type_que": tag_rep["pred_type_que"],
                "span_doc": [eval(i) for sen_span in tag_rep["hpsg_list_doc"] for i in sen_span],
                "type_doc": [i for sen in tag_rep["pred_type_doc"] for i in sen],
                "head_doc": [int(i) for sen_head in tag_rep["pred_head_doc"] for i in sen_head],
                "token_doc": [token for sen_token in tag_rep['doc_tokens'] for token in sen_token],
                "token_que": tag_rep['que_tokens']
            }
            all_dqtag_data.append(dqtag_data)

        simple_nlp = SimpleNlp()
        examples = []
        for entry in tqdm(input_data, ncols=50, desc="reading examples:"):
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]

                sen_texts = simple_nlp.nlp(paragraph_text)
                sen_list = []

                for sen_ix, sent in enumerate(sen_texts.sents):
                    sent_tokens = []
                    prev_is_whitespace = True
                    for c in sent.string:
                        if self._is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                sent_tokens.append(c)
                            else:
                                sent_tokens[-1] += c
                            prev_is_whitespace = False
                    sen_list.append((sen_ix, sent_tokens))

                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if self._is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                cnt_token = 0
                new_sen_list = []
                flag = False
                tmp_token = ""
                for sen_ix, sent_tokens in sen_list:
                    new_sent_tokens = []
                    for tok_ix, token in enumerate(sent_tokens):
                        if tok_ix == 0 and flag:
                            token = tmp_token + token
                            flag = False
                            tmp_token = ""
                            assert token == doc_tokens[cnt_token]

                        if token != doc_tokens[cnt_token]:
                            assert tok_ix == len(sent_tokens) - 1
                            tmp_token = token
                            flag = True
                        else:
                            assert token == doc_tokens[cnt_token]
                            new_sent_tokens.append(token)

                            cnt_token += 1
                    new_sen_list.append(new_sent_tokens)

                # double check
                cnt_token = 0
                for sent_tokens in new_sen_list:
                    new_sent_tokens = []
                    for tok_ix, token in enumerate(sent_tokens):
                        assert token == doc_tokens[cnt_token]
                        new_sent_tokens.append(token)
                        cnt_token += 1

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    dqtag = all_dqtag_data[qas_id_to_tag_idx_map[qas_id]]
                    assert dqtag["qas_id"] == qas_id

                    span_doc = dqtag["span_doc"]
                    head_doc = dqtag["head_doc"]
                    type_doc = dqtag["type_doc"]
                    assert len(span_doc) == len(head_doc) == len(type_doc) == cnt_token
                    # reconstruct into sentences
                    new_span_doc = []
                    new_head_doc = []
                    new_type_doc = []
                    cnt = 0
                    for sent_tokens in new_sen_list:
                        new_span_sen = []
                        new_head_sen = []
                        new_type_sen = []
                        for _ in sent_tokens:
                            new_span_sen.append(span_doc[cnt])
                            new_head_sen.append(head_doc[cnt])
                            new_type_sen.append(type_doc[cnt])
                            cnt += 1
                        new_span_doc.append(new_span_sen)
                        new_head_doc.append(new_head_sen)
                        new_type_doc.append(new_type_sen)

                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if datatype == "train":
                        is_impossible = qa["is_impossible"]

                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                self._whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                print("Could not find answer: '%s' vs. '%s'",
                                      actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""
                    datable("qas_id", qas_id)
                    datable("question_text", question_text)
                    datable("doc_tokens", doc_tokens)
                    datable("orig_answer_text", orig_answer_text)
                    datable("start_position", start_position)
                    datable("end_position", end_position)
                    datable("is_impossible", is_impossible)
                    datable("que_heads", dqtag["head_que"])
                    datable("que_types", dqtag["type_que"])
                    datable("que_span", dqtag["span_que"])
                    datable("doc_heads", new_head_doc)
                    datable("doc_types", new_type_doc)
                    datable("doc_span", new_span_doc)
                    datable("token_doc", dqtag["token_doc"])
                    datable("token_que", dqtag["token_que"])

        return datable

    def _is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _read_train(self, path, tag_path=None):
        return self._read_data(path=path, tag_path=tag_path, datatype="train")

    def _read_dev(self, path, tag_path=None):
        return self._read_data(path=path, tag_path=tag_path, datatype="dev")

    def _read_test(self, path, tag_path=None):
        return None

    def read_all(self):
        return self._read_train(self.train_path, self.train_tag_path), \
               self._read_dev(self.dev_path, self.dev_tag_path), \
               self._read_test(self.test_path, self.test_tag_path)

    def read_vocab(self):
        return None


if __name__ == "__main__":
    reader = Squad2SubsetReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0_subset/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
