from cogktr.data.reader.conll2003_reader import Conll2003Reader
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from tqdm import tqdm
from cogktr.utils.tokenizer_utils import SequenceBertTokenizer


class Conll2003Processor:
    def __init__(self, plm, max_token_len, vocab):
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = SequenceBertTokenizer.from_pretrained(plm)

    def process(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence, ner_label in tqdm(zip(data['sentence'], data['ner_labels']), total=len(data['sentence'])):
            input_ids, token_type_ids, attention_mask, head_flag_matrix = self.tokenizer.encode(word_sequence=sentence,
                                                                                                max_length=self.max_token_len,
                                                                                                add_special_tokens=True)
            ner_label = ner_label + (self.max_token_len - len(ner_label)) * self.max_token_len * ["<pad>"]
            ner_label = ner_label[:self.max_token_len]
            datable("input_ids", input_ids)
            datable("attention_mask", attention_mask)
            datable("head_flag_matrix", head_flag_matrix)
            datable("ner_label", self.vocab["ner_label_vocab"].labels2ids(ner_label))
        return DataTableSet(datable)


if __name__ == "__main__":
    reader = Conll2003Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sequence_labeling/conll2003/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    processor = Conll2003Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
    train_dataset = processor.process(train_data)
    dev_dataset = processor.process(dev_data)
    test_dataset = processor.process(test_data)
    print("end")
