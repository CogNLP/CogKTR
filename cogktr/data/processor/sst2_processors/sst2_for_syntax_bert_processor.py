from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
import numpy as np
import copy

transformers.logging.set_verbosity_error()  # set transformers logging level


class Sst2ForSyntaxBertProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, n_mask, vocab):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.n_mask = n_mask
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)

    def process_train(self, data, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")
        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            input_tokens = []
            input_ids = []
            attention_masks = np.zeros((self.n_mask * 3, self.max_token_len, self.max_token_len))
            segment_ids = []
            wordpiece2word_loc_list = []
            word2wordpiece_loc_list = []
            raw_words = enhanced_dict[sentence]["syntax"]["words"]
            deprel_labels = enhanced_dict[sentence]["syntax"]["deprel_labels"]
            head_labels = enhanced_dict[sentence]["syntax"]["head_labels"]
            indexes = enhanced_dict[sentence]["syntax"]["indexes"]

            words_len = len(raw_words)
            words = copy.deepcopy(raw_words)
            words.insert(0, '[CLS]')
            words.append('[SEP]')

            for i, word in enumerate(words):
                token = self.tokenizer.tokenize(word)
                input_tokens.extend(token)
                for j in range(len(token)):
                    if word == '[CLS]':
                        wordpiece2word_loc_list.append([0])
                    elif word == '[SEP]':
                        wordpiece2word_loc_list.append([i - 2])
                    else:
                        wordpiece2word_loc_list.append([i - 1])
            word2wordpiece_loc_list = [[] for _ in range(words_len)]
            for i, wordpiece2word_loc in enumerate(wordpiece2word_loc_list):
                word2wordpiece_loc_list[wordpiece2word_loc[0]].append(i)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            for depth in range(1, self.n_mask + 1):  # n_mask是垂直单向搜索深度，需要遍历各个深度

                for i in range(words_len):  # 遍历每一个原始word(不含cls和sep)
                    par = int(indexes[i])  # 当前word序列编号，从1开始编号
                    flag = 0  # 在depth范围内，1表示搜索到根了，0表示没有搜到跟
                    for j in range(depth):
                        if (par == 0):
                            flag = 1
                            break
                        par = int(head_labels[par - 1])

                    if flag == 0 and par != 0:  # 没有搜到根的时候，进行以下步骤
                        word_idx_0 = int(indexes[i]) - 1  # 当前节点的下标
                        word_idx_1 = par - 1  # 父亲节点的下标
                        token_idx_0 = word2wordpiece_loc_list[word_idx_0]  # 当前节点的wordpiece下标列表
                        token_idx_1 = word2wordpiece_loc_list[word_idx_1]  # 父亲节点的wordpiece下标列表
                        idx_0 = np.array(token_idx_0)
                        idx_1 = np.array(token_idx_1)
                        attention_masks[depth - 1, idx_0[0]:idx_0[-1] + 1, idx_1[0]:idx_1[-1] + 1] = 1  # parent mask
                        attention_masks[depth - 1 + self.n_mask, idx_1[0]:idx_1[-1] + 1,
                        idx_0[0]:idx_0[-1] + 1] = 1  # child_mask

            for i in range(words_len):
                for j in range(words_len):
                    if int(indexes[i]) == int(indexes[j]):  # 如果是单词自己，就直接跳过
                        continue

                    depthi = 0  # 深度，root也算节点，root深度算0
                    par = int(indexes[i])
                    while par != 0:
                        depthi += 1
                        par = int(head_labels[par - 1])

                    depthj = 0  # 深度，root也算节点，root深度算0
                    par = int(indexes[j])
                    while par != 0:
                        depthj += 1
                        par = int(head_labels[par - 1])

                    dist = 0
                    cur_i = int(indexes[i])
                    cur_j = int(indexes[j])
                    if depthi > depthj:
                        dist += (depthi - depthj)
                        for i in range(depthi - depthj):
                            cur_i = int(head_labels[cur_i - 1])
                    if depthi < depthj:
                        dist += (depthj - depthi)
                        for i in range(depthj - depthi):
                            cur_j = int(head_labels[cur_j - 1])
                    if cur_i == cur_j:
                        continue
                    while cur_i != cur_j:
                        dist += 2
                        cur_i = int(head_labels[cur_i - 1])
                        cur_j = int(head_labels[cur_j - 1])

                    if dist - 2 < self.n_mask:
                        word_idx_0 = int(indexes[i]) - 1
                        word_idx_1 = int(indexes[j]) - 1
                        token_idx_0 = word2wordpiece_loc_list[word_idx_0]
                        token_idx_1 = word2wordpiece_loc_list[word_idx_1]
                        idx_0 = np.array(token_idx_0)
                        idx_1 = np.array(token_idx_1)
                        attention_masks[dist - 2 + self.n_mask * 2, idx_0[0]:idx_0[-1] + 1, idx_1[0]:idx_1[-1] + 1] = 1
                        attention_masks[dist - 2 + self.n_mask * 2, idx_1[0]:idx_1[-1] + 1, idx_0[0]:idx_0[-1] + 1] = 1

            attention_masks_f = np.zeros((self.n_mask, self.max_token_len, self.max_token_len))
            attention_masks_f[:, 0:len(input_tokens), 0:len(input_tokens)] = 1
            attention_masks = np.concatenate([attention_masks, attention_masks_f], axis=0)

            segment_ids = [0] * len(input_ids)

            input_ids = input_ids[0:self.max_token_len]
            segment_ids = segment_ids[0:self.max_token_len]
            wordpiece2word_loc_list = wordpiece2word_loc_list[0:self.max_token_len]

            input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
            segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
            wordpiece2word_loc_list += [0 for _ in range(self.max_token_len - len(wordpiece2word_loc_list))]

            datable("input_ids", input_ids)
            datable("attention_masks", attention_masks)
            datable("segment_ids", segment_ids)
            datable("labels", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_dev(self, data, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")
        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            input_tokens = []
            input_ids = []
            attention_masks = np.zeros((self.n_mask * 3, self.max_token_len, self.max_token_len))
            segment_ids = []
            wordpiece2word_loc_list = []
            word2wordpiece_loc_list = []
            raw_words = enhanced_dict[sentence]["syntax"]["words"]
            deprel_labels = enhanced_dict[sentence]["syntax"]["deprel_labels"]
            head_labels = enhanced_dict[sentence]["syntax"]["head_labels"]
            indexes = enhanced_dict[sentence]["syntax"]["indexes"]

            words_len = len(raw_words)
            words = copy.deepcopy(raw_words)
            words.insert(0, '[CLS]')
            words.append('[SEP]')

            for i, word in enumerate(words):
                token = self.tokenizer.tokenize(word)
                input_tokens.extend(token)
                for j in range(len(token)):
                    if word == '[CLS]':
                        wordpiece2word_loc_list.append([0])
                    elif word == '[SEP]':
                        wordpiece2word_loc_list.append([i - 2])
                    else:
                        wordpiece2word_loc_list.append([i - 1])
            word2wordpiece_loc_list = [[] for _ in range(words_len)]
            for i, wordpiece2word_loc in enumerate(wordpiece2word_loc_list):
                word2wordpiece_loc_list[wordpiece2word_loc[0]].append(i)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            for depth in range(1, self.n_mask + 1):  # n_mask是垂直单向搜索深度，需要遍历各个深度

                for i in range(words_len):  # 遍历每一个原始word(不含cls和sep)
                    par = int(indexes[i])  # 当前word序列编号，从1开始编号
                    flag = 0  # 在depth范围内，1表示搜索到根了，0表示没有搜到跟
                    for j in range(depth):
                        if (par == 0):
                            flag = 1
                            break
                        par = int(head_labels[par - 1])

                    if flag == 0 and par != 0:  # 没有搜到根的时候，进行以下步骤
                        word_idx_0 = int(indexes[i]) - 1  # 当前节点的下标
                        word_idx_1 = par - 1  # 父亲节点的下标
                        token_idx_0 = word2wordpiece_loc_list[word_idx_0]  # 当前节点的wordpiece下标列表
                        token_idx_1 = word2wordpiece_loc_list[word_idx_1]  # 父亲节点的wordpiece下标列表
                        idx_0 = np.array(token_idx_0)
                        idx_1 = np.array(token_idx_1)
                        attention_masks[depth - 1, idx_0[0]:idx_0[-1] + 1, idx_1[0]:idx_1[-1] + 1] = 1  # parent mask
                        attention_masks[depth - 1 + self.n_mask, idx_1[0]:idx_1[-1] + 1,
                        idx_0[0]:idx_0[-1] + 1] = 1  # child_mask

            for i in range(words_len):
                for j in range(words_len):
                    if int(indexes[i]) == int(indexes[j]):  # 如果是单词自己，就直接跳过
                        continue

                    depthi = 0  # 深度，root也算节点，root深度算0
                    par = int(indexes[i])
                    while par != 0:
                        depthi += 1
                        par = int(head_labels[par - 1])

                    depthj = 0  # 深度，root也算节点，root深度算0
                    par = int(indexes[j])
                    while par != 0:
                        depthj += 1
                        par = int(head_labels[par - 1])

                    dist = 0
                    cur_i = int(indexes[i])
                    cur_j = int(indexes[j])
                    if depthi > depthj:
                        dist += (depthi - depthj)
                        for i in range(depthi - depthj):
                            cur_i = int(head_labels[cur_i - 1])
                    if depthi < depthj:
                        dist += (depthj - depthi)
                        for i in range(depthj - depthi):
                            cur_j = int(head_labels[cur_j - 1])
                    if cur_i == cur_j:
                        continue
                    while cur_i != cur_j:
                        dist += 2
                        cur_i = int(head_labels[cur_i - 1])
                        cur_j = int(head_labels[cur_j - 1])

                    if dist - 2 < self.n_mask:
                        word_idx_0 = int(indexes[i]) - 1
                        word_idx_1 = int(indexes[j]) - 1
                        token_idx_0 = word2wordpiece_loc_list[word_idx_0]
                        token_idx_1 = word2wordpiece_loc_list[word_idx_1]
                        idx_0 = np.array(token_idx_0)
                        idx_1 = np.array(token_idx_1)
                        attention_masks[dist - 2 + self.n_mask * 2, idx_0[0]:idx_0[-1] + 1, idx_1[0]:idx_1[-1] + 1] = 1
                        attention_masks[dist - 2 + self.n_mask * 2, idx_1[0]:idx_1[-1] + 1, idx_0[0]:idx_0[-1] + 1] = 1

            attention_masks_f = np.zeros((self.n_mask, self.max_token_len, self.max_token_len))
            attention_masks_f[:, 0:len(input_tokens), 0:len(input_tokens)] = 1
            attention_masks = np.concatenate([attention_masks, attention_masks_f], axis=0)

            segment_ids = [0] * len(input_ids)

            input_ids = input_ids[0:self.max_token_len]
            segment_ids = segment_ids[0:self.max_token_len]
            wordpiece2word_loc_list = wordpiece2word_loc_list[0:self.max_token_len]

            input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
            segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
            wordpiece2word_loc_list += [0 for _ in range(self.max_token_len - len(wordpiece2word_loc_list))]

            datable("input_ids", input_ids)
            datable("attention_masks", attention_masks)
            datable("segment_ids", segment_ids)
            datable("labels", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_test(self, data, enhanced_dict=None):
        datable = DataTable()
        print("Processing data...")
        for sentence in tqdm(data['sentence'], total=len(data['sentence'])):
            input_tokens = []
            input_ids = []
            attention_masks = np.zeros((self.n_mask * 3, self.max_token_len, self.max_token_len))
            segment_ids = []
            wordpiece2word_loc_list = []
            word2wordpiece_loc_list = []
            raw_words = enhanced_dict[sentence]["syntax"]["words"]
            deprel_labels = enhanced_dict[sentence]["syntax"]["deprel_labels"]
            head_labels = enhanced_dict[sentence]["syntax"]["head_labels"]
            indexes = enhanced_dict[sentence]["syntax"]["indexes"]

            words_len = len(raw_words)
            words = copy.deepcopy(raw_words)
            words.insert(0, '[CLS]')
            words.append('[SEP]')

            for i, word in enumerate(words):
                token = self.tokenizer.tokenize(word)
                input_tokens.extend(token)
                for j in range(len(token)):
                    if word == '[CLS]':
                        wordpiece2word_loc_list.append([0])
                    elif word == '[SEP]':
                        wordpiece2word_loc_list.append([i - 2])
                    else:
                        wordpiece2word_loc_list.append([i - 1])
            word2wordpiece_loc_list = [[] for _ in range(words_len)]
            for i, wordpiece2word_loc in enumerate(wordpiece2word_loc_list):
                word2wordpiece_loc_list[wordpiece2word_loc[0]].append(i)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            for depth in range(1, self.n_mask + 1):  # n_mask是垂直单向搜索深度，需要遍历各个深度

                for i in range(words_len):  # 遍历每一个原始word(不含cls和sep)
                    par = int(indexes[i])  # 当前word序列编号，从1开始编号
                    flag = 0  # 在depth范围内，1表示搜索到根了，0表示没有搜到跟
                    for j in range(depth):
                        if (par == 0):
                            flag = 1
                            break
                        par = int(head_labels[par - 1])

                    if flag == 0 and par != 0:  # 没有搜到根的时候，进行以下步骤
                        word_idx_0 = int(indexes[i]) - 1  # 当前节点的下标
                        word_idx_1 = par - 1  # 父亲节点的下标
                        token_idx_0 = word2wordpiece_loc_list[word_idx_0]  # 当前节点的wordpiece下标列表
                        token_idx_1 = word2wordpiece_loc_list[word_idx_1]  # 父亲节点的wordpiece下标列表
                        idx_0 = np.array(token_idx_0)
                        idx_1 = np.array(token_idx_1)
                        attention_masks[depth - 1, idx_0[0]:idx_0[-1] + 1, idx_1[0]:idx_1[-1] + 1] = 1  # parent mask
                        attention_masks[depth - 1 + self.n_mask, idx_1[0]:idx_1[-1] + 1,
                        idx_0[0]:idx_0[-1] + 1] = 1  # child_mask

            for i in range(words_len):
                for j in range(words_len):
                    if int(indexes[i]) == int(indexes[j]):  # 如果是单词自己，就直接跳过
                        continue

                    depthi = 0  # 深度，root也算节点，root深度算0
                    par = int(indexes[i])
                    while par != 0:
                        depthi += 1
                        par = int(head_labels[par - 1])

                    depthj = 0  # 深度，root也算节点，root深度算0
                    par = int(indexes[j])
                    while par != 0:
                        depthj += 1
                        par = int(head_labels[par - 1])

                    dist = 0
                    cur_i = int(indexes[i])
                    cur_j = int(indexes[j])
                    if depthi > depthj:
                        dist += (depthi - depthj)
                        for i in range(depthi - depthj):
                            cur_i = int(head_labels[cur_i - 1])
                    if depthi < depthj:
                        dist += (depthj - depthi)
                        for i in range(depthj - depthi):
                            cur_j = int(head_labels[cur_j - 1])
                    if cur_i == cur_j:
                        continue
                    while cur_i != cur_j:
                        dist += 2
                        cur_i = int(head_labels[cur_i - 1])
                        cur_j = int(head_labels[cur_j - 1])

                    if dist - 2 < self.n_mask:
                        word_idx_0 = int(indexes[i]) - 1
                        word_idx_1 = int(indexes[j]) - 1
                        token_idx_0 = word2wordpiece_loc_list[word_idx_0]
                        token_idx_1 = word2wordpiece_loc_list[word_idx_1]
                        idx_0 = np.array(token_idx_0)
                        idx_1 = np.array(token_idx_1)
                        attention_masks[dist - 2 + self.n_mask * 2, idx_0[0]:idx_0[-1] + 1, idx_1[0]:idx_1[-1] + 1] = 1
                        attention_masks[dist - 2 + self.n_mask * 2, idx_1[0]:idx_1[-1] + 1, idx_0[0]:idx_0[-1] + 1] = 1

            attention_masks_f = np.zeros((self.n_mask, self.max_token_len, self.max_token_len))
            attention_masks_f[:, 0:len(input_tokens), 0:len(input_tokens)] = 1
            attention_masks = np.concatenate([attention_masks, attention_masks_f], axis=0)

            segment_ids = [0] * len(input_ids)

            input_ids = input_ids[0:self.max_token_len]
            segment_ids = segment_ids[0:self.max_token_len]
            wordpiece2word_loc_list = wordpiece2word_loc_list[0:self.max_token_len]

            input_ids += [0 for _ in range(self.max_token_len - len(input_ids))]
            segment_ids += [0 for _ in range(self.max_token_len - len(segment_ids))]
            wordpiece2word_loc_list += [0 for _ in range(self.max_token_len - len(wordpiece2word_loc_list))]

            datable("input_ids", input_ids)
            datable("attention_masks", attention_masks)
            datable("segment_ids", segment_ids)
        return DataTableSet(datable)
