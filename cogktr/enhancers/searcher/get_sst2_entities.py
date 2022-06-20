from cogktr import *
from tqdm import tqdm
from cogktr.enhancers.searcher.kilt_searcher import KnowledgeSource

reader = Sst2Reader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/raw_data")
train_data, dev_data, test_data = reader.read_all()

ks = KnowledgeSource()
wikidata_id_list = []

print("Processing Test Data...\n")
for sentence in tqdm(test_data.datas['sentence']):
    linker = WikipediaLinker(tool="tagme")
    link_list = linker.link(sentence)
    for entity in link_list:
        try:
            wikidata_id = ks.get_page_by_id(entity['id'])["wikidata_info"]["wikidata_id"]
            if wikidata_id not in wikidata_id_list:
                wikidata_id_list.append(wikidata_id)
        except:
            pass

print("Processing Train Data...\n")
for sentence in tqdm(train_data.datas['sentence']):
    linker = WikipediaLinker(tool="tagme")
    link_list = linker.link(sentence)
    for entity in link_list:
        try:
            wikidata_id = ks.get_page_by_id(entity['id'])["wikidata_info"]["wikidata_id"]
            if wikidata_id not in wikidata_id_list:
                wikidata_id_list.append(wikidata_id)
        except:
            pass

print("Processing Dec Data...\n")
for sentence in tqdm(dev_data.datas['sentence']):
    linker = WikipediaLinker(tool="tagme")
    link_list = linker.link(sentence)
    for entity in link_list:
        try:
            wikidata_id = ks.get_page_by_id(entity['id'])["wikidata_info"]["wikidata_id"]
            if wikidata_id not in wikidata_id_list:
                wikidata_id_list.append(wikidata_id)
        except:
            pass

with open('sst2_wikidata_id_list.txt', 'w') as f:
    for item in wikidata_id_list:
        f.write("%s\n" % item)



