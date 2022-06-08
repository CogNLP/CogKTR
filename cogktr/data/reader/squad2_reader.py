import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.io_utils import load_json
from cogktr.utils.download_utils import Downloader


class Squad2Reader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_squad2_raw_data(raw_data_path)
        self.train_file = 'train-v2.0.json'
        self.dev_file = 'dev-v2.0.json'
        self.test_file = ''
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)

    def _read_train(self, path):
        datable = DataTable()
        raw_data_dict = load_json(path)
        data = raw_data_dict["data"]
        for line in data:
            title = line["title"]
            paragraphs = line["paragraphs"]
            for paragraph in paragraphs:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]

                    start_position,end_position = 0,0

                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True

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

                    if start_position_character is not None and not is_impossible:
                        start_position = char_to_word_offset[start_position_character]
                        end_position = char_to_word_offset[
                            min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
                        ]
                    datable("qas_id",qas_id)
                    datable("is_impossible",is_impossible)
                    datable("question_text",question_text)
                    datable("context_text",context_text)
                    datable("answer_text",answer_text)
                    datable("start_position",start_position)
                    datable("end_position",end_position)
                    datable("doc_tokens",doc_tokens)
                    datable("title",title)
        return datable

    def _read_dev(self, path):
        return self._read_train(path)

    def _read_test(self, path):
        return None

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        return None

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

if __name__ == "__main__":
    reader = Squad2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
