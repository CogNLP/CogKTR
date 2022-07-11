from cogktr.data.reader.base_reader import BaseReader
from cogktr.utils.download_utils import Downloader
import os
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.data.datable import DataTable
import json


class OpenBookQAReader(BaseReader):
    def __init__(self, raw_data_path, use_cache=True):
        super(OpenBookQAReader, self).__init__()
        self.raw_data_path = raw_data_path
        self.use_cache = use_cache
        downloader = Downloader()
        downloader.download_openbookqa_raw_data(raw_data_path)
        self.train_file = 'train.jsonl'
        self.dev_file = 'dev.jsonl'
        self.test_file = 'test.jsonl'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()
        self.addition = {}

    def _read_train(self, path):
        return self._read_data(path)

    def _read_dev(self, path):
        return self._read_data(path)

    def _read_data(self, path):
        datable = DataTable()
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            json_line = json.loads(line)
            output_dict = convert_qajson_to_entailment(json_line)
            datable("output_dict", output_dict)
        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path)


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(qa_json: dict):
    question_text = qa_json["question"]["stem"]
    choices = qa_json["question"]["choices"]
    for choice in choices:
        choice_text = choice["text"]
        statement = question_text + ' ' + choice_text
        create_output_dict(qa_json, statement, choice["label"] == qa_json.get("answerKey", "A"))

    return qa_json


# Create the output json dictionary from the input json, premise and hypothesis statement
def create_output_dict(input_json: dict, statement: str, label: bool) -> dict:
    if "statements" not in input_json:
        input_json["statements"] = []
    input_json["statements"].append({"label": label, "statement": statement})
    return input_json


if __name__ == '__main__':
    reader = OpenBookQAReader(
        raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data"
    )
    train_data, dev_data = reader.read_all()
