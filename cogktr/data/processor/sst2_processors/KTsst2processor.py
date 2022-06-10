from cogktr.data.processor.sst2_processors import SST2Processor
from cogktr.enhancers.linker.wikipedia_linker import WikipediaLinker
from cogktr.enhancers.searcher.wikipedia_searcher import WikipediaSearcher
from cogktr.data.reader.sst2_reader import SST2Reader
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from tqdm import tqdm


class KTSST2Processor(SST2Processor):
    def __init__(self, plm, max_token_len, vocab, knowledge_path):
        super().__init__(plm, max_token_len, vocab)
        self.knowledge_path = knowledge_path

    def process_train(self, data):
        datable = DataTable()
        print("Integrating knowledge text and processing data...")
        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            sentence = self.integrate_kt(sentence, self.knowledge_path)
            token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
                                          max_length=self.max_token_len)
            datable("token", token)
            datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    @staticmethod
    def integrate_kt(sentence, knowledge_path, linker_tool="tagme", searcher_tool="blink"):
        linker = WikipediaLinker(tool=linker_tool)
        searcher = WikipediaSearcher(tool=searcher_tool, path=knowledge_path)
        entity_ids = [entity["id"] for entity in linker.link(sentence).values()]
        for entity_id in entity_ids:
            sentence = sentence + searcher.search(entity_id)[entity_id]['desc']
        return sentence


if __name__ == "__main__":
    # sentence = KTSST2Processor.integrate_kt(sentence="Bert likes reading in the library.",
    #                                        knowledge_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikipedia_desc/entity.jsonl")
    reader = SST2Reader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    processor = KTSST2Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab,
                                knowledge_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikipedia_desc/entity.jsonl")
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
