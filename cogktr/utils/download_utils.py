import os


class Downloader:
    def __init__(self, url="http://210.75.240.138:8080/download/CogNLP_data/CogKTR_data"):
        self.url = url

    def _download_data(self, output_file, dataset_list, zip_name):
        if not os.path.exists(output_file):
            raise FileExistsError("{} does not exist".format(output_file))
        download_flag = False
        for data_name in dataset_list:
            if not os.path.exists(os.path.join(output_file, data_name)):
                download_flag = True
        file_url = os.path.join(self.url, zip_name)
        if download_flag:
            os.system("wget -P {} {}".format(output_file, file_url))
            os.system("cd {} && unzip {}".format(output_file, zip_name))

    def download_conll2003_raw_data(self, output_file):
        dataset_list = ["train.txt", "valid.txt", "test.txt", "metadata"]
        zip_name = "RAW_CONLL2003.zip"
        self._download_data(output_file=output_file,
                            dataset_list=dataset_list,
                            zip_name=zip_name)

    def download_qnli_raw_data(self, output_file):
        dataset_list = ["train.tsv", "dev.tsv", "test.tsv"]
        zip_name = "RAW_QNLI.zip"
        self._download_data(output_file=output_file,
                            dataset_list=dataset_list,
                            zip_name=zip_name)

    def download_squad2_raw_data(self, output_file):
        dataset_list = ["train-v2.0.json", "dev-v2.0.json"]
        zip_name = "RAW_SQUAD2.zip"
        self._download_data(output_file=output_file,
                            dataset_list=dataset_list,
                            zip_name=zip_name)

    def download_sst2_raw_data(self, output_file):
        dataset_list = ["train.tsv", "dev.tsv", "test.tsv", "original"]
        zip_name = "RAW_SST2.zip"
        self._download_data(output_file=output_file,
                            dataset_list=dataset_list,
                            zip_name=zip_name)

    def download_stsb_raw_data(self, output_file):
        dataset_list = ["train.tsv", "dev.tsv", "test.tsv", "original", "LICENSE.txt"]
        zip_name = "RAW_STSB.zip"
        self._download_data(output_file=output_file,
                            dataset_list=dataset_list,
                            zip_name=zip_name)

    def download_conll2005_srl_subset_raw_data(self, output_file):
        dataset_list = ["train.json", "dev.json", "brown-test.json", "wsj-test.json"]
        zip_name = "RAW_CONLL2005_SRL_SUBSET.zip"
        self._download_data(output_file=output_file,
                            dataset_list=dataset_list,
                            zip_name=zip_name)


if __name__ == "__main__":
    downloader = Downloader()
    downloader.download_conll2003_raw_data("/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2003/raw_data")
    downloader.download_qnli_raw_data("/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
    downloader.download_squad2_raw_data("/data/mentianyi/code/CogKTR/datapath/reading_comprehension/SQuAD2.0/raw_data")
    downloader.download_sst2_raw_data("/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    downloader.download_stsb_raw_data("/data/mentianyi/code/CogKTR/datapath/sentence_pair/STS_B/raw_data")
    downloader.download_conll2005_srl_subset_raw_data("/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2005_srl_subset/raw_data")
