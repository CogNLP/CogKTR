from cogktr.data.reader.sst2reader import SST2Reader
from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm


class SST2Processor:
    def __init__(self,plm,max_token_len):
        self.plm=plm
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.max_token_len=max_token_len

    def process(self,data):
        datable = DataTable()
        print("Processing data...")
        for sentence, label in tqdm(zip(data['sentence'], data['label']),total=len(data['sentence'])):
            token=self.tokenizer.encode(text=sentence,truncation=True,padding="max_length",add_special_tokens=True,max_length=self.max_token_len)
            datable("token",token)
            datable("label", label)
        return DataTableSet(datable)

if __name__=="__main__":
    reader=SST2Reader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/SST_2/raw_data")
    train_data,dev_data,test_data=reader.read_all()
    processor=SST2Processor(plm="bert-base-cased",max_token_len=128)
    train_dataset=processor.process(train_data)
    dev_dataset=processor.process(dev_data)
    test_dataset=processor.process(test_data)
    print("end")