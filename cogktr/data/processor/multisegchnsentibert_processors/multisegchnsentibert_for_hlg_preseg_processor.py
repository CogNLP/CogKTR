from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from tqdm import tqdm
import transformers
import numpy as np
import torch
from cogktr.data.processor.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()  # set transformers logging level


class MultisegchnsentibertForHLGPresegProcessor(BaseProcessor):
    def __init__(self, max_token_len, vocab):
        super().__init__()
        self.max_token_len = max_token_len
        self.vocab = vocab

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence, thulac_word_length, nlpir_word_length, hanlp_word_length, label in tqdm(
                zip(data['sentence'], data['thulac_word_length'], data['nlpir_word_length'], data['hanlp_word_length'],
                    data['label']), total=len(data['sentence'])):
            if len(sentence) > self.max_token_len - 2:
                sentence = sentence[:self.max_token_len]

            start_end_toolid_list = []
            for toolid, word_length_list in enumerate([thulac_word_length, nlpir_word_length, hanlp_word_length]):
                start_loc = 0
                for word_length in word_length_list:
                    start = start_loc
                    end = word_length + start_loc
                    start_end_toolid = (start, end, toolid)
                    start_loc = start_loc + word_length
                    start_end_toolid_list.append(start_end_toolid)

            start_end = sorted(list(set(map(lambda x: x[:2], start_end_toolid_list))))

            graph_character, graph_word = [], []
            for word_index, (start, end) in enumerate(start_end):
                for character_index in range(start, end):
                    if character_index >= self.max_token_len - 2:
                        break
                    if word_index >= self.max_token_len:
                        continue
                    graph_character.append(character_index)
                    graph_word.append(word_index)

            character_len = len(sentence)
            word_len = len(graph_word)

            graph_sw_word, graph_sw_sentence = [], []
            for start_end_toolid in start_end_toolid_list:
                if len(graph_sw_word) >= self.max_token_len:
                    break
                start_end_item = start_end_toolid[:2]
                toolid_item = start_end_toolid[2]
                word_index = start_end.index(start_end_item)
                if word_index >= self.max_token_len:
                    continue
                graph_sw_word.append(word_index)
                graph_sw_sentence.append(toolid_item)

            input_ids = sentence + [0] * (self.max_token_len - character_len)
            input_mask = [1] * character_len + [0] * (self.max_token_len - character_len)

            character_mask = np.zeros(self.max_token_len)
            word_mask = np.zeros(self.max_token_len)
            sentence_mask = np.zeros(self.max_token_len)

            max_graph_character_id = int(max(graph_character))
            character_mask[:max_graph_character_id] = np.ones(max_graph_character_id)
            max_graph_word_id = int(max(graph_word))
            word_mask[:max_graph_word_id] = np.ones(max_graph_word_id)
            sentence_mask[:3] = np.array([1, 1, 1])

            datable("input_ids", input_ids)
            datable("graph_character", graph_character)
            datable("graph_word", graph_word)
            datable("graph_sw_word", graph_sw_word)
            datable("graph_sw_sentence", graph_sw_sentence)
            datable("character_mask", character_mask)
            datable("word_mask", word_mask)
            datable("sentence_mask", sentence_mask)
            datable("input_mask", input_mask)
            datable("label", self.vocab["label_vocab"].label2id(label))

        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)

    def _collate(self, batch):
        batch_size = len(batch)
        input_ids = []
        batch_c2w = []
        batch_w2s = []
        character_mask = []
        word_mask = []
        sentence_mask = []
        input_mask = []
        label = []
        for i in range(batch_size):
            input_ids.append(batch[i]["input_ids"])
            g_c = batch[i]["graph_character"]
            g_w = batch[i]["graph_word"]
            g_sw_w = batch[i]["graph_sw_word"]
            g_sw_s = batch[i]["graph_sw_sentence"]
            character_mask.append(torch.tensor(batch[i]["character_mask"]))
            word_mask.append(torch.tensor(batch[i]["word_mask"]))
            sentence_mask.append(torch.tensor(batch[i]["sentence_mask"]))
            input_mask.append(batch[i]["input_mask"])
            label.append(batch[i]["label"])

            c2w = torch.stack([g_c, g_w])
            c2w_graph_size = torch.Size([self.max_token_len, self.max_token_len])
            c2w_graph_item_values = torch.ones(g_c.size(0))
            c2w_graph = torch.sparse.IntTensor(c2w, c2w_graph_item_values, c2w_graph_size)

            w2s = torch.stack([g_sw_w, g_sw_s])
            w2s_graph_size = torch.Size([self.max_token_len, self.max_token_len])
            w2s_graph_item_values = torch.ones(g_sw_w.size(0))
            w2s_graph = torch.sparse.IntTensor(w2s, w2s_graph_item_values, w2s_graph_size)

            batch_c2w.append(c2w_graph.to_dense())
            batch_w2s.append(w2s_graph.to_dense())

        batch = {}
        batch["input_ids"] = torch.stack(input_ids)
        batch["c2w"] = torch.stack(batch_c2w)
        batch["w2s"] = torch.stack(batch_w2s)
        batch["character_mask"] = torch.stack(character_mask).to(torch.float32)
        batch["word_mask"] = torch.stack(word_mask).to(torch.float32)
        batch["sentence_mask"] = torch.stack(sentence_mask).to(torch.float32)
        batch["input_mask"] = torch.stack(input_mask)
        batch["label"] = torch.stack(label)

        return batch

    def train_collate(self, batch):
        return self._collate(batch)

    def dev_collate(self, batch):
        return self._collate(batch)

    def test_collate(self, batch):
        return self._collate(batch)


if __name__ == "__main__":
    from cogktr.data.reader.multisegchnsentibert_reader import MultisegchnsentibertReader

    reader = MultisegchnsentibertReader(
        raw_data_path="/data/mentianyi/code/CogKTR/datapath/text_classification/MultiSegChnSentiBERT/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = MultisegchnsentibertForHLGPresegProcessor(max_token_len=128, vocab=vocab)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
