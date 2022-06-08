from cogktr.data.reader.sst2_reader import Sst2Reader
from cogktr.enhancers import Enhancer
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class Sst24KtembProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, enhancer):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.enhancer = enhancer
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def process_train(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
                                          max_length=self.max_token_len)
            datable("wikipedia", self.enhancer.get_knowledge(sentence)["wikipedia"])
            datable("token", token)
            datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_dev(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
                                          max_length=self.max_token_len)
            datable("wikipedia", self.enhancer.get_knowledge(sentence)["wikipedia"])
            datable("token", token)
            datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_test(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence in tqdm(zip(data['sentence']), total=len(data['sentence'])):
            token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
                                          max_length=self.max_token_len)
            datable("wikipedia", self.enhancer.get_knowledge(sentence)["wikipedia"])
            datable("token", token)
        return DataTableSet(datable)


if __name__ == "__main__":
    reader = Sst2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    enhancer = Enhancer(return_entity_desc=True)
    enhancer.set_config(
        WikipediaSearcherPath="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia/raw_data/entity.jsonl",
        WikipediaEmbedderPath="/data/mentianyi/code/CogKTR/datapath/knowledge_graph/wikipedia2vec/raw_data/enwiki_20180420_win10_100d.pkl")
    processor = Sst24KtembProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab, enhancer=enhancer)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
