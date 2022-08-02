import os
from cogktr.data.reader.base_reader import BaseReader
from cogktr.data.datable import DataTable
from cogktr.utils.vocab_utils import Vocabulary
from cogktr.utils.download_utils import Downloader
import json


class LamaReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        downloader = Downloader()
        downloader.download_lama_raw_data(raw_data_path)
        self.object_vocab = Vocabulary()

    def _read_data(self, dataset_name):
        if dataset_name.lower() not in ["google_re", "trex", "squad", "conceptnet"]:
            raise ValueError("No dataset named {}".format(dataset_name))
        datable = DataTable()
        if dataset_name.lower() == "google_re":
            for file_name in os.listdir(os.path.join(self.raw_data_path, "Google_RE")):
                with open(os.path.join(self.raw_data_path, "Google_RE", file_name)) as file:
                    lines = file.readlines()
                for line in lines:
                    line_dict = json.loads(line)
                    masked_sent = "".join(line_dict["masked_sentences"])
                    object = line_dict["obj_label"]
                    datable("masked_sent", masked_sent)
                    datable("object", object)
                    self.object_vocab.add(object)

        if dataset_name.lower() == "trex":
            for file_name in os.listdir(os.path.join(self.raw_data_path, "TREx")):
                with open(os.path.join(self.raw_data_path, "TREx", file_name)) as file:
                    lines = file.readlines()
                for line in lines:
                    sent_list = json.loads(line)["evidences"]
                    for sent_dict in sent_list:
                        masked_sent = sent_dict["masked_sentence"]
                        object = sent_dict["obj_surface"]
                        datable("masked_sent", masked_sent)
                        datable("object", object)
                        self.object_vocab.add(object)
        return datable

    def read_all(self, dataset_name=None, isSplit=False):
        if isSplit:
            return self._read_data(dataset_name).split(0.8, 0.1, 0.1)
        else:
            return self._read_data(dataset_name)

    def read_vocab(self):
        self.object_vocab.create()
        return {"object_vocab": self.object_vocab}


if __name__ == "__main__":
    reader = LamaReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/masked_language_model/LAMA/raw_data")
    test_data = reader.read_all(dataset_name="google_re")
    vocab = reader.read_vocab()
    print("end")

# import os
# from cogktr.data.reader.base_reader import BaseReader
# from cogktr.data.datable import DataTable
# from cogktr.utils.vocab_utils import Vocabulary
# from cogktr.utils.download_utils import Downloader
# import json
#
#
# class LamaReader(BaseReader):
#     def __init__(self, raw_data_path):
#         super().__init__()
#         self.raw_data_path = raw_data_path
#         downloader = Downloader()
#         downloader.download_lama_raw_data(raw_data_path)
#         self.object_vocab = Vocabulary()
#
#     def _read_data(self, dataset_name):
#         if dataset_name.lower() not in ["google_re", "trex", "squad", "conceptnet"]:
#             raise ValueError("No dataset named {}".format(dataset_name))
#         datable = DataTable()
#         if dataset_name.lower() == "google_re":
#             for file_name in os.listdir(os.path.join(self.raw_data_path, "Google_RE")):
#                 with open(os.path.join(self.raw_data_path, "Google_RE", file_name)) as file:
#                     lines = file.readlines()
#                 for line in lines:
#                     line_dict = json.loads(line)
#                     masked_sent = "".join(line_dict["masked_sentences"])
#                     object = line_dict["obj_label"]
#                     datable("masked_sent", masked_sent)
#                     datable("object", object)
#                     self.object_vocab.add(object)
#
#         if dataset_name.lower() == "trex":
#             for file_name in os.listdir(os.path.join(self.raw_data_path, "TREx")):
#                 with open(os.path.join(self.raw_data_path, "TREx", file_name)) as file:
#                     lines = file.readlines()
#                 for line in lines:
#                     sent_list = json.loads(line)["evidences"]
#                     for sent_dict in sent_list:
#                         masked_sent = sent_dict["masked_sentence"]
#                         object = sent_dict["obj_surface"]
#                         datable("masked_sent", masked_sent)
#                         datable("object", object)
#                         self.object_vocab.add(object)
#         return datable
#
#     def read_all(self, dataset_name=None, isSplit=True):
#         if isSplit:
#             return self._read_data(dataset_name).split(0.8, 0.1, 0.1)
#         else:
#             return self._read_data(dataset_name)
#
#     def read_vocab(self):
#         return {}
#
#
# if __name__ == "__main__":
#     reader = LamaReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/masked_language_model/LAMA/raw_data")
#     train_data, dev_data, test_data = reader.read_all(dataset_name="trex", isSplit=True)
#     vocab = reader.read_vocab()
#     print("end")
