from cogktr.data.datable import DataTable
from cogktr.data.datableset import DataTableSet
from transformers import RobertaTokenizer,RobertaForMaskedLM
from tqdm import tqdm
import transformers
from cogktr.data.processor.base_processor import BaseProcessor
from collections import OrderedDict
import numpy as np
import torch

transformers.logging.set_verbosity_error()  # set transformers logging level


class OpenBookQAForQagnnProcessor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab, device,debug=False):
        super().__init__()
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = RobertaTokenizer.from_pretrained(plm)
        self.debug = debug
        self.lm_model = RobertaForMaskedLMwithLoss.from_pretrained(plm)
        self.device = device
        self.lm_model.to(device)
        self.lm_model.eval()
        self.max_node_num=200


    def _process(self, data, enhanced_data_dict=None):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        num_choices = 4
        for ii in tqdm(range(len(data)//num_choices)):
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            special_tokens_mask_list = []
            answerKey = data["answerKey"][ii * num_choices]

            edge_index, edge_type = [], []
            adj_lengths_list = []
            concept_ids_list = []
            node_type_ids_list = []
            node_scores_list = []

            for jj in range(num_choices):
                kk = num_choices * ii + jj
                id,stem,answer_text,key,statement,answerKey = data[kk]
                data_dict = enhanced_data_dict[(statement,answer_text)]["interaction"]
                cid2score = get_LM_score(data_dict["concept"],
                                         data_dict["concept_names"],
                                         statement,
                                         self.tokenizer,
                                         self.lm_model,
                                         self.device)
                adj,concepts,qm,am = data_dict["adj"],data_dict["concept"],data_dict["qmask"],data_dict["amask"]
                assert len(concepts) == len(set(concepts))
                qam = qm | am
                assert qam[0] == True
                F_start = False
                for TF in qam:
                    if TF == False:
                        F_start = True
                    else:
                        assert F_start == False
                num_concept = min(len(concepts),
                                  self.max_node_num - 1) + 1  # this is the final number of nodes including contextnode but excluding PAD
                adj_lengths_orig = len(concepts)
                adj_lengths = num_concept

                concept_ids = np.full((self.max_node_num,),1)
                node_type_ids = np.full((self.max_node_num,),2)
                node_scores = np.zeros((self.max_node_num,1))

                # Prepare nodes
                concepts = concepts[:num_concept - 1]
                concept_ids[1:num_concept] = np.array(concepts + 1)  # To accomodate contextnode, original concept_ids incremented by 1
                concept_ids[0] = 0  # this is the "concept_id" for contextnode

                # Prepare node scores
                if (cid2score is not None):
                    for _j_ in range(num_concept):
                        _cid = int(concept_ids[ _j_]) - 1
                        assert _cid in cid2score
                        node_scores[_j_, 0] = np.array(cid2score[_cid])

                # Prepare node types
                node_type_ids[0] = 3  # contextnode
                node_type_ids[1:num_concept][np.array(qm, dtype=np.bool)[:num_concept - 1]] = 0
                node_type_ids[1:num_concept][np.array(am, dtype=np.bool)[:num_concept - 1]] = 1

                # Load adj
                ij = np.array(adj.row, dtype=np.int64)  # (num_matrix_entries, ), where each entry is coordinate
                k = np.array(adj.col, dtype=np.int64)  # (num_matrix_entries, ), where each entry is coordinate
                n_node = adj.shape[1]
                half_n_rel = adj.shape[0] // n_node
                i, j = ij // n_node, ij % n_node

                # Prepare edges
                i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
                extra_i, extra_j, extra_k = [], [], []
                for _coord, q_tf in enumerate(qm):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if q_tf:
                        extra_i.append(0)  # rel from contextnode to question concept
                        extra_j.append(0)  # contextnode coordinate
                        extra_k.append(_new_coord)  # question concept coordinate
                for _coord, a_tf in enumerate(am):
                    _new_coord = _coord + 1
                    if _new_coord > num_concept:
                        break
                    if a_tf:
                        extra_i.append(1)  # rel from contextnode to answer concept
                        extra_j.append(0)  # contextnode coordinate
                        extra_k.append(_new_coord)  # answer concept coordinate

                half_n_rel += 2  # should be 19 now
                if len(extra_i) > 0:
                    i = np.concatenate([i, np.array(extra_i)], axis=0)
                    j = np.concatenate([j, np.array(extra_j)], axis=0)
                    k = np.concatenate([k, np.array(extra_k)], axis=0)

                mask = (j < self.max_node_num) & (k < self.max_node_num)
                i, j, k = i[mask], j[mask], k[mask]
                i, j, k = np.concatenate((i, i + half_n_rel), 0), np.concatenate((j, k), 0), np.concatenate((k, j),
                                                                                             0)  # add inverse relations
                edge_index.append(torch.tensor(np.stack([j, k], axis=0)))  # each entry is [2, E]
                edge_type.append(torch.tensor(i))  # each entry is [E, ]
                adj_lengths_list.append(adj_lengths)
                concept_ids_list.append(concept_ids.tolist())
                node_type_ids_list.append(node_type_ids.tolist())
                node_scores_list.append(node_scores.tolist())

                tokenized_data = self.tokenizer.encode_plus(text=stem, text_pair=answer_text,
                                                            truncation='longest_first',
                                                            padding="max_length",
                                                            add_special_tokens=True,
                                                            return_token_type_ids=True,
                                                            return_special_tokens_mask=True,
                                                            max_length=self.max_token_len)
                input_ids = tokenized_data["input_ids"]
                attention_mask = tokenized_data["attention_mask"]
                token_type_ids = tokenized_data["token_type_ids"]
                special_tokens_mask = tokenized_data["special_tokens_mask"]
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)
                special_tokens_mask_list.append(special_tokens_mask)

            # stack the choice features together to create one sample
            # datable("input_ids",np.array(input_ids_list,dtype=int))
            # datable("attention_mask", np.array(attention_mask_list,dtype=int))
            # datable("token_type_ids", np.array(token_type_ids_list,dtype=int))
            # datable("special_tokens_mask", np.array(special_tokens_mask_list,dtype=int))
            # datable("adj_lengths",np.array(adj_lengths_list,int))
            # datable("concept_ids",np.array(concept_ids_list,int))
            # datable("node_tpye_ids",np.array(node_type_ids_list,int))
            # datable("node_scores",np.array(node_scores_list,float))
            datable("input_ids",input_ids_list)
            datable("attention_mask", attention_mask_list)
            datable("token_type_ids", token_type_ids_list)
            datable("special_tokens_mask", special_tokens_mask_list)
            datable("adj_length", adj_lengths_list)
            datable("concept_id", concept_ids_list)
            datable("node_type_id",node_type_ids_list)
            datable("node_score",node_scores_list)
            datable("edge_index",edge_index)
            datable("edge_type",edge_type)
            datable("label",self.vocab["label_vocab"].label2id(answerKey))
        datable.not2torch.add("edge_index")
        datable.not2torch.add("edge_type")
        return DataTableSet(datable)

    def process_train(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_dev(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

    def process_test(self, data, enhanced_data_dict=None):
        return self._process(data, enhanced_data_dict)

class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs

def get_LM_score(cids,concept_names,question,tokenizer,lm_model,device):
    cids = cids[:]
    cids = np.insert(cids,0,-1)
    concept_names.insert(0,"")
    sents, scores = [], []
    for idx,cid in enumerate(cids):
        if cid==-1:
            sent = question.lower()
        else:
            sent = '{} {}.'.format(question.lower(), ' '.join(concept_names[idx]))
        sent = tokenizer.encode(sent, add_special_tokens=True)
        sents.append(sent)
    n_cids = len(cids)
    cur_idx = 0
    batch_size = 50
    while cur_idx < n_cids:
        #Prepare batch
        input_ids = sents[cur_idx: cur_idx+batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [tokenizer.pad_token_id] * (max_len-len(seq))
            input_ids[j] = seq
        input_ids = torch.tensor(input_ids).to(device) #[B, seqlen]
        mask = (input_ids!=1).long() #[B, seq_len]
        #Get LM score
        with torch.no_grad():
            outputs = lm_model(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            loss = outputs[0] #[B, ]
            _scores = list(-loss.detach().cpu().numpy()) #list of float
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1])) #score: from high to low
    return cid2score


