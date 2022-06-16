# This code is moved from here:
# https://github.com/DevSinghSachan/syntax-augmented-bert


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy.stats as stats
from pytorch_transformers import modeling_bert

EMB_INIT_RANGE = 1.0
MAX_LEN = 100
MAX_POSITION = 10

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

INFINITY_NUMBER = 1e12

# hard-coded mappings from fields to ids
POS_TO_ID = {PAD_TOKEN: 0,
             UNK_TOKEN: 1,
             'NNP': 2,
             'NN': 3,
             'IN': 4,
             'DT': 5,
             ',': 6,
             'JJ': 7,
             'NNS': 8,
             'VBD': 9,
             'CD': 10,
             'CC': 11,
             '.': 12,
             'RB': 13,
             'VBN': 14,
             'PRP': 15,
             'TO': 16,
             'VB': 17,
             'VBG': 18,
             'VBZ': 19,
             'PRP$': 20,
             ':': 21,
             'POS': 22,
             '\'\'': 23,
             '``': 24,
             '-RRB-': 25,
             '-LRB-': 26,
             'VBP': 27,
             'MD': 28,
             'NNPS': 29,
             'WP': 30,
             'WDT': 31,
             'WRB': 32,
             'RP': 33,
             'JJR': 34,
             'JJS': 35,
             '$': 36,
             'FW': 37,
             'RBR': 38,
             'SYM': 39,
             'EX': 40,
             'RBS': 41,
             'WP$': 42,
             'PDT': 43,
             'LS': 44,
             'UH': 45,
             '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                'acl': 2,
                'acl:relcl': 3,
                'acomp': 4,
                'advcl': 5,
                'advmod': 6,
                'amod': 7,
                'appos': 8,
                'aux': 9,
                'auxpass': 10,
                'aux:pass': 10,
                'case': 11,
                'cc': 12,
                'cc:preconj': 13,
                'ccomp': 14,
                'compound': 15,
                'compound:prt': 16,
                'conj': 17,
                'cop': 18,
                'csubj': 19,
                'csubjpass': 20,
                'csubj:pass': 20,
                'dep': 21,
                'det': 22,
                'det:predet': 23,
                'discourse': 24,
                'dobj': 25,
                'erased': 26,
                'expl': 27,
                'iobj': 28,
                'mark': 29,
                'mwe': 30,
                'neg': 31,
                'nmod': 32,
                'nmod:npmod': 33,
                'nmod:poss': 34,
                'nmod:tmod': 35,
                'nn': 36,
                'npadvmod': 37,
                'nsubj': 38,
                'nsubjpass': 39,
                'nsubj:pass': 39,
                'num': 40,
                'number': 41,
                'nummod': 42,
                'parataxis': 43,
                'pcomp': 44,
                'pobj': 45,
                'poss': 46,
                'possessive': 47,
                'preconj': 48,
                'predet': 49,
                'prep': 50,
                'prt': 51,
                'punct': 52,
                'quantmod': 53,
                'rcmod': 54,
                'root': 55,
                'tmod': 56,
                'vmod': 57,
                'xcomp': 58,
                'fixed': 59,
                'obj': 60,
                'obl': 61,
                'obl:npmod': 62,
                'obl:tmod': 63,
                'orphan': 64,
                'flat': 65,
                'list': 66,
                'vocative': 67,
                'reparandum': 68,
                'goeswith': 69,
                'flat:foreign': 70,
                'subtokens': 71,
                'special_rel': 72}

UDEPREL_TO_ID = {PAD_TOKEN: 0,
                 UNK_TOKEN: 1,
                 'acl': 2,
                 'advcl': 3,
                 'advmod': 4,
                 'amod': 5,
                 'appos': 6,
                 'aux': 7,
                 'case': 8,
                 'cc': 9,
                 'ccomp': 10,
                 'clf': 11,
                 'compound': 12,
                 'conj': 13,
                 'cop': 14,
                 'csubj': 15,
                 'dep': 16,
                 'det': 17,
                 'discourse': 18,
                 'dislocated': 19,
                 'expl': 20,
                 'fixed': 21,
                 'flat': 22,
                 'goeswith': 23,
                 'iobj': 24,
                 'list': 25,
                 'mark': 26,
                 'nmod': 27,
                 'nsubj': 28,
                 'nummod': 29,
                 'obj': 30,
                 'obl': 31,
                 'orphan': 32,
                 'parataxis': 33,
                 'punct': 34,
                 'reparandum': 35,
                 'root': 36,
                 'vocative': 37,
                 'xcomp': 38,
                 'subtokens': 39,
                 'special_rel': 40
                 }

# These come from TACRED Relation Extraction Dataset
TACRED_SPECIAL_ENTITY_SET = {'OBJ-RELIGION',
                             'SUBJ-PERSON',
                             'OBJ-IDEOLOGY',
                             'OBJ-NUMBER',
                             'SUBJ-ORGANIZATION',
                             'OBJ-TITLE',
                             'OBJ-MISC',
                             'OBJ-PERSON',
                             'OBJ-STATE_OR_PROVINCE',
                             'OBJ-ORGANIZATION',
                             'OBJ-DATE',
                             'OBJ-LOCATION',
                             'OBJ-CRIMINAL_CHARGE',
                             'OBJ-URL',
                             'OBJ-CITY',
                             'OBJ-DURATION',
                             'OBJ-NATIONALITY',
                             'OBJ-CAUSE_OF_DEATH',
                             'OBJ-COUNTRY'}

TACRED_LABEL_TO_ID = {'no_relation': 0,
                      'per:title': 1,
                      'org:top_members/employees': 2,
                      'per:employee_of': 3,
                      'org:alternate_names': 4,
                      'org:country_of_headquarters': 5,
                      'per:countries_of_residence': 6,
                      'org:city_of_headquarters': 7,
                      'per:cities_of_residence': 8,
                      'per:age': 9,
                      'per:stateorprovinces_of_residence': 10,
                      'per:origin': 11,
                      'org:subsidiaries': 12,
                      'org:parents': 13,
                      'per:spouse': 14,
                      'org:stateorprovince_of_headquarters': 15,
                      'per:children': 16,
                      'per:other_family': 17,
                      'per:alternate_names': 18,
                      'org:members': 19,
                      'per:siblings': 20,
                      'per:schools_attended': 21,
                      'per:parents': 22,
                      'per:date_of_death': 23,
                      'org:member_of': 24,
                      'org:founded_by': 25,
                      'org:website': 26,
                      'per:cause_of_death': 27,
                      'org:political/religious_affiliation': 28,
                      'org:founded': 29,
                      'per:city_of_death': 30,
                      'org:shareholders': 31,
                      'org:number_of_employees/members': 32,
                      'per:date_of_birth': 33,
                      'per:city_of_birth': 34,
                      'per:charges': 35,
                      'per:stateorprovince_of_death': 36,
                      'per:religion': 37,
                      'per:stateorprovince_of_birth': 38,
                      'per:country_of_birth': 39,
                      'org:dissolved': 40,
                      'per:country_of_death': 41}

TACRED_CLASS_WEIGHTS = [8.0900e-01, 3.5861e-02, 2.7744e-02, 2.2371e-02, 1.1861e-02, 6.8698e-03,
                        6.5322e-03, 5.6074e-03, 5.4900e-03, 5.7249e-03, 4.8588e-03, 4.7707e-03,
                        4.3450e-03, 4.1982e-03, 3.7872e-03, 3.3615e-03, 3.0973e-03, 2.6276e-03,
                        1.5266e-03, 2.4954e-03, 2.4221e-03, 2.1872e-03, 2.2312e-03, 1.9670e-03,
                        1.7909e-03, 1.8202e-03, 1.6294e-03, 1.7175e-03, 1.5413e-03, 1.3358e-03,
                        1.1890e-03, 1.1156e-03, 1.1009e-03, 9.2478e-04, 9.5414e-04, 1.0569e-03,
                        7.1928e-04, 7.7799e-04, 5.5781e-04, 4.1102e-04, 3.3762e-04, 8.8075e-05]

OntoNotes_NER_LABEL_TO_ID = {'PAD_TOKEN': 0,
                             'B-ORG': 1,
                             'I-ORG': 2,
                             'B-PERSON': 3,
                             'I-PERSON': 4,
                             'B-GPE': 5,
                             'I-GPE': 6,
                             'B-DATE': 7,
                             'I-DATE': 8,
                             'B-CARDINAL': 9,
                             'I-CARDINAL': 10,
                             'B-MONEY': 11,
                             'I-MONEY': 12,
                             'B-PERCENT': 13,
                             'I-PERCENT': 14,
                             'B-WORK_OF_ART': 15,
                             'I-WORK_OF_ART': 16,
                             'B-ORDINAL': 17,
                             'I-ORDINAL': 18,
                             'B-EVENT': 19,
                             'I-EVENT': 20,
                             'B-LOC': 21,
                             'I-LOC': 22,
                             'B-QUANTITY': 23,
                             'I-QUANTITY': 24,
                             'B-TIME': 25,
                             'I-TIME': 26,
                             'B-FAC': 27,
                             'I-FAC': 28,
                             'B-LAW': 29,
                             'I-LAW': 30,
                             'B-PRODUCT': 31,
                             'I-PRODUCT': 32,
                             'B-NORP': 33,
                             'I-NORP': 34,
                             'B-LANGUAGE': 35,
                             'I-LANGUAGE': 36,
                             'O': 37}

OntoNotes_SRL_LABEL_TO_ID = {'PAD_TOKEN': 0,
                             'B-ARG0': 1,
                             'I-ARG0': 2,
                             'B-V': 3,
                             'B-ARG1': 4,
                             'I-ARG1': 5,
                             'B-ARG2': 6,
                             'I-ARG2': 7,
                             'B-ARGM-TMP': 8,
                             'I-ARGM-TMP': 9,
                             'B-ARGM-NEG': 10,
                             'I-ARGM-NEG': 11,
                             'B-ARGM-ADJ': 12,
                             'I-ARGM-ADJ': 13,
                             'B-ARGM-LOC': 14,
                             'I-ARGM-LOC': 15,
                             'B-ARGM-MOD': 16,
                             'I-ARGM-MOD': 17,
                             'B-ARG3': 18,
                             'I-ARG3': 19,
                             'B-R-ARG2': 20,
                             'I-R-ARG2': 21,
                             'B-ARGM-ADV': 22,
                             'I-ARGM-ADV': 23,
                             'B-R-ARG1': 24,
                             'I-R-ARG1': 25,
                             'B-ARGM-DIS': 26,
                             'I-ARGM-DIS': 27,
                             'B-ARGM-MNR': 28,
                             'I-ARGM-MNR': 29,
                             'B-R-ARG0': 30,
                             'I-R-ARG0': 31,
                             'B-ARGM-PRP': 32,
                             'I-ARGM-PRP': 33,
                             'B-ARGM-PRD': 34,
                             'I-ARGM-PRD': 35,
                             'B-ARGM-GOL': 36,
                             'I-ARGM-GOL': 37,
                             'B-ARG4': 38,
                             'I-ARG4': 39,
                             'B-ARGM-CAU': 40,
                             'I-ARGM-CAU': 41,
                             'B-R-ARGM-LOC': 42,
                             'I-R-ARGM-LOC': 43,
                             'B-ARGM-EXT': 44,
                             'I-ARGM-EXT': 45,
                             'B-ARGM-DIR': 46,
                             'I-ARGM-DIR': 47,
                             'B-ARGM-COM': 48,
                             'I-ARGM-COM': 49,
                             'B-C-ARGM-EXT': 50,
                             'I-C-ARGM-EXT': 51,
                             'B-C-ARGM-COM': 52,
                             'I-C-ARGM-COM': 53,
                             'B-C-ARGM-ADV': 54,
                             'I-C-ARGM-ADV': 55,
                             'B-C-ARGM-CAU': 56,
                             'I-C-ARGM-CAU': 57,
                             'B-R-ARGM-PRD': 58,
                             'I-R-ARGM-PRD': 59,
                             'B-C-ARGM-TMP': 60,
                             'I-C-ARGM-TMP': 61,
                             'B-ARG5': 62,
                             'I-ARG5': 63,
                             'B-ARGM-PNC': 64,
                             'I-ARGM-PNC': 65,
                             'B-C-ARG4': 66,
                             'I-C-ARG4': 67,
                             'B-R-ARGM-MNR': 68,
                             'I-R-ARGM-MNR': 69,
                             'B-C-ARG2': 70,
                             'I-C-ARG2': 71,
                             'B-C-ARG3': 72,
                             'I-C-ARG3': 73,
                             'B-C-ARG1': 74,
                             'I-C-ARG1': 75,
                             'B-R-ARGM-CAU': 76,
                             'I-R-ARGM-CAU': 77,
                             'I-R-ARGM-ADV': 78,
                             'B-R-ARGM-PRP': 79,
                             'I-R-ARGM-PRP': 80,
                             'B-R-ARGM-EXT': 81,
                             'I-R-ARGM-EXT': 82,
                             'B-C-ARGM-MOD': 83,
                             'I-C-ARGM-MOD': 84,
                             'B-R-ARGM-PNC': 85,
                             'I-R-ARGM-PNC': 86,
                             'B-C-ARGM-DIS': 87,
                             'I-C-ARGM-DIS': 88,
                             'B-C-ARGM-ADJ': 89,
                             'I-C-ARGM-ADJ': 90,
                             'B-C-ARGM-DIR': 91,
                             'I-C-ARGM-DIR': 92,
                             'B-C-ARG0': 93,
                             'I-C-ARG0': 94,
                             'B-C-ARGM-LOC': 95,
                             'I-C-ARGM-LOC': 96,
                             'B-R-ARGM-COM': 97,
                             'I-R-ARGM-COM': 98,
                             'B-C-ARGM-MNR': 99,
                             'I-C-ARGM-MNR': 100,
                             'B-R-ARGM-DIR': 101,
                             'I-R-ARGM-DIR': 102,
                             'B-R-ARGM-TMP': 103,
                             'I-R-ARGM-TMP': 104,
                             'B-ARGM-REC': 105,
                             'I-ARGM-REC': 106,
                             'B-R-ARG4': 107,
                             'I-R-ARG4': 108,
                             'B-R-ARG3': 109,
                             'I-R-ARG3': 110,
                             'B-C-ARGM-NEG': 111,
                             'I-C-ARGM-NEG': 112,
                             'B-R-ARGM-GOL': 113,
                             'I-R-ARGM-GOL': 114,
                             'B-C-ARGM-PRP': 115,
                             'I-C-ARGM-PRP': 116,
                             'B-C-ARGM-DSP': 117,
                             'I-C-ARGM-DSP': 118,
                             'B-ARGM-LVB': 119,
                             'B-R-ARGM-ADV': 120,
                             'B-ARGA': 121,
                             'I-ARGA': 122,
                             'B-ARGM-PRR': 123,
                             'B-ARGM-DSP': 124,
                             'I-ARGM-DSP': 125,
                             'B-ARGM-PRX': 126,
                             'B-R-ARG5': 127,
                             'B-R-ARGM-MOD': 128,
                             'O': 129}

CoNLL2005_SRL_LABEL_TO_ID = {'PAD_TOKEN': 0,
                             'B-A0': 1,
                             'B-A1': 2,
                             'B-A2': 3,
                             'B-A3': 4,
                             'B-A4': 5,
                             'B-A5': 6,
                             'B-AA': 7,
                             'B-AM': 8,
                             'B-AM-ADV': 9,
                             'B-AM-CAU': 10,
                             'B-AM-DIR': 11,
                             'B-AM-DIS': 12,
                             'B-AM-EXT': 13,
                             'B-AM-LOC': 14,
                             'B-AM-MNR': 15,
                             'B-AM-MOD': 16,
                             'B-AM-NEG': 17,
                             'B-AM-PNC': 18,
                             'B-AM-PRD': 19,
                             'B-AM-REC': 20,
                             'B-AM-TM': 21,
                             'B-AM-TMP': 22,
                             'B-C-A0': 23,
                             'B-C-A1': 24,
                             'B-C-A2': 25,
                             'B-C-A3': 26,
                             'B-C-A4': 27,
                             'B-C-A5': 28,
                             'B-C-AM-ADV': 29,
                             'B-C-AM-CAU': 30,
                             'B-C-AM-DIR': 31,
                             'B-C-AM-DIS': 32,
                             'B-C-AM-EXT': 33,
                             'B-C-AM-LOC': 34,
                             'B-C-AM-MNR': 35,
                             'B-C-AM-NEG': 36,
                             'B-C-AM-PNC': 37,
                             'B-C-AM-TMP': 38,
                             'B-C-V': 39,
                             'B-R-A0': 40,
                             'B-R-A1': 41,
                             'B-R-A2': 42,
                             'B-R-A3': 43,
                             'B-R-A4': 44,
                             'B-R-AA': 45,
                             'B-R-AM-ADV': 46,
                             'B-R-AM-CAU': 47,
                             'B-R-AM-DIR': 48,
                             'B-R-AM-EXT': 49,
                             'B-R-AM-LOC': 50,
                             'B-R-AM-MNR': 51,
                             'B-R-AM-PNC': 52,
                             'B-R-AM-TMP': 53,
                             'B-V': 54,
                             'I-A0': 55,
                             'I-A1': 56,
                             'I-A2': 57,
                             'I-A3': 58,
                             'I-A4': 59,
                             'I-A5': 60,
                             'I-AA': 61,
                             'I-AM': 62,
                             'I-AM-ADV': 63,
                             'I-AM-CAU': 64,
                             'I-AM-DIR': 65,
                             'I-AM-DIS': 66,
                             'I-AM-EXT': 67,
                             'I-AM-LOC': 68,
                             'I-AM-MNR': 69,
                             'I-AM-MOD': 70,
                             'I-AM-NEG': 71,
                             'I-AM-PNC': 72,
                             'I-AM-PRD': 73,
                             'I-AM-REC': 74,
                             'I-AM-TM': 75,
                             'I-AM-TMP': 76,
                             'I-C-A0': 77,
                             'I-C-A1': 78,
                             'I-C-A2': 79,
                             'I-C-A3': 80,
                             'I-C-A4': 81,
                             'I-C-A5': 82,
                             'I-C-AM-ADV': 83,
                             'I-C-AM-CAU': 84,
                             'I-C-AM-DIR': 85,
                             'I-C-AM-DIS': 86,
                             'I-C-AM-EXT': 87,
                             'I-C-AM-LOC': 88,
                             'I-C-AM-MNR': 89,
                             'I-C-AM-PNC': 90,
                             'I-C-AM-TMP': 91,
                             'I-C-V': 92,
                             'I-R-A0': 93,
                             'I-R-A1': 94,
                             'I-R-A2': 95,
                             'I-R-A3': 96,
                             'I-R-A4': 97,
                             'I-R-AM-ADV': 98,
                             'I-R-AM-DIR': 99,
                             'I-R-AM-EXT': 100,
                             'I-R-AM-LOC': 101,
                             'I-R-AM-MNR': 102,
                             'I-R-AM-PNC': 103,
                             'I-R-AM-TMP': 104,
                             'I-V': 105,
                             'O': 106}


class HighwayGateLayer(nn.Module):
    def __init__(self, in_out_size, bias=True):
        super(HighwayGateLayer, self).__init__()
        self.transform = nn.Linear(in_out_size, in_out_size, bias=bias)

    def forward(self, x, y):
        out_transform = torch.sigmoid(self.transform(x))
        return out_transform * x + (1 - out_transform) * y


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32):
    """Outputs random values from a truncated normal distribution.
    The generated values follow a normal distribution with specified mean
    and standard deviation, except that values whose magnitude is more
    than 2 standard deviations from the mean are dropped and re-picked.
    API from: https://www.tensorflow.org/api_docs/python/tf/truncated_normal
    """
    lower = -2 * stddev + mean
    upper = 2 * stddev + mean
    X = stats.truncnorm((lower - mean) / stddev,
                        (upper - mean) / stddev,
                        loc=mean,
                        scale=stddev)
    values = X.rvs(size=shape)
    return torch.from_numpy(values.astype(dtype))


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_size, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_size)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a truncated normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters using Truncated Normal Initializer (default in Tensorflow)
        """
        self.weight.data = truncated_normal(shape=(self.num_embeddings,
                                                   self.embedding_dim),
                                            stddev=1.0 / math.sqrt(self.embedding_dim))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class GELU(nn.Module):
    def __init__(self, inplace=False):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.gelu(input, inplace=self.inplace)


class GNNClassifier(nn.Module):
    """ A wrapper classifier for GNNRelationModel. """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.syntax_encoder = GNNRelationModel(config)
        self.classifier = nn.Linear(config.syntax['hidden_size'], config.num_labels)

    def resize_token_embeddings(self, new_num_tokens=None):
        self.syntax_encoder.resize_token_embeddings(new_num_tokens)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, dep_head=None, dep_rel=None, wp_rows=None, align_sizes=None,
                seq_len=None, subj_pos=None, obj_pos=None):
        pooled_output, sequence_output = self.syntax_encoder(input_ids,
                                                             attention_mask,
                                                             dep_head,
                                                             dep_rel,
                                                             seq_len)
        logits = self.classifier(pooled_output)
        return logits

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func: from_pretrained`` class method.
        """
        return


class GNNRelationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create embedding layers
        if config.model_type == "late_fusion":
            in_dim = config.hidden_size
        else:  # for pre-trained models such as "joint_fusion", or for randomly initialized ones like "gat"
            self.emb = ScaledEmbedding(config.vocab_size,
                                       config.syntax['emb_size'],
                                       padding_idx=PAD_ID)
            in_dim = config.syntax['emb_size']
            self.input_dropout = nn.Dropout(config.syntax['input_dropout'])

            if config.syntax['embed_position']:
                self.embed_pos = ScaledEmbedding(config.max_position_embeddings,
                                                 config.syntax['emb_size'])

        if config.syntax['use_dep_rel']:
            self.rel_emb = ScaledEmbedding(len(DEPREL_TO_ID),
                                           int(config.hidden_size / config.num_attention_heads),
                                           padding_idx=DEPREL_TO_ID['<PAD>'])

        # LSTM layer
        if config.syntax['contextual_rnn']:
            self.rnn = nn.LSTM(config.syntax['emb_size'],
                               config.syntax['rnn_hidden'],
                               config.syntax['rnn_layers'],
                               batch_first=True,
                               dropout=config.syntax['rnn_dropout'],
                               bidirectional=True)
            in_dim = config.syntax['rnn_hidden'] * 2
            self.rnn_dropout = nn.Dropout(config.syntax['rnn_dropout'])  # use on last layer output

        # Graph Attention layer
        if config.use_syntax:
            self.syntax_encoder = eval(config.syntax['syntax_encoder'])(config)

            if config.model_type == 'late_fusion' and config.syntax['late_fusion_gated_connection']:
                self.gate = eval(config.syntax['late_fusion_gate'])(config.syntax['hidden_size'])

            out_dim = config.syntax['hidden_size']
        else:
            out_dim = in_dim

        if config.syntax['use_subj_obj']:
            out_dim *= 3

        # output MLP layers
        layers = [nn.Linear(out_dim,
                            config.syntax['hidden_size']),
                  nn.Tanh()]
        for _ in range(config.syntax['mlp_layers'] - 1):
            layers += [nn.Linear(config.syntax['hidden_size'],
                                 config.syntax['hidden_size']),
                       nn.Tanh()]
        self.out_mlp = nn.Sequential(*layers)
        self.pool_mask, self.subj_mask, self.obj_mask = (None, None, None)

    def resize_token_embeddings(self, new_num_tokens=None):
        if new_num_tokens is None:
            return

        old_num_tokens, old_embedding_dim = self.emb.weight.size()
        if old_num_tokens == new_num_tokens:
            return

        # Build new embeddings
        new_embeddings = ScaledEmbedding(new_num_tokens,
                                         old_embedding_dim,
                                         padding_idx=PAD_ID)
        new_embeddings.to(self.emb.weight.device)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = self.emb.weight.data[:num_tokens_to_copy, :]
        self.emb = new_embeddings

    def encode_with_rnn(self, rnn_inputs, seq_lens):
        batch_size = rnn_inputs.size(0)
        h0, c0 = rnn_zero_state(batch_size,
                                self.config.syntax['rnn_hidden'],
                                self.config.syntax['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs,
                                                       seq_lens,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs,
                                         (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs,
                                                          batch_first=True)
        return rnn_outputs

    def forward(self, input_ids_or_bert_hidden, adj=None, dep_rel_matrix=None,
                wp_seq_lengths=None):

        if self.config.model_type == 'late_fusion':
            if self.config.syntax['finetune_bert']:
                embeddings = input_ids_or_bert_hidden
            else:
                embeddings = Variable(input_ids_or_bert_hidden.data)
        else:
            embeddings = self.emb(input_ids_or_bert_hidden)

            if self.config.syntax['embed_position']:
                seq_length = input_ids_or_bert_hidden.size(1)
                position_ids = torch.arange(seq_length,
                                            dtype=torch.long,
                                            device=input_ids_or_bert_hidden.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids_or_bert_hidden)
                position_embeddings = self.embed_pos(position_ids)
                embeddings += position_embeddings

            embeddings = self.input_dropout(embeddings)

            if self.config.syntax['contextual_rnn']:
                embeddings = self.rnn_dropout(self.encode_with_rnn(embeddings,
                                                                   wp_seq_lengths))
        syntax_inputs = embeddings

        dep_rel_emb = None
        if self.config.syntax['use_dep_rel']:
            dep_rel_emb = self.rel_emb(dep_rel_matrix)

        if self.config.use_syntax:
            attention_mask = adj.clone().detach().unsqueeze(1)
            # attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0

            h = self.syntax_encoder(syntax_inputs,
                                    attention_mask,
                                    dep_rel_emb)
            if self.config.model_type == 'late_fusion' and self.config.syntax['late_fusion_gated_connection']:
                h = self.gate(syntax_inputs,
                              h)
        else:
            h = syntax_inputs

        return h


class GATEncoder(nn.Module):
    def __init__(self, config):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(config.syntax['num_layers']):
            layer = GATEncoderLayer(config)
            self.layers.append(layer)
        self.ln = nn.LayerNorm(config.hidden_size,
                               eps=1e-6)

    def forward(self, e, attention_mask, dep_rel_matrix=None):
        for layer in self.layers:
            e = layer(e,
                      attention_mask,
                      dep_rel_matrix)
        e = self.ln(e)
        return e


class GATEncoderLayer(nn.Module):
    def __init__(self, config):
        super(GATEncoderLayer, self).__init__()
        self.config = config
        self.syntax_attention = RelationalBertSelfAttention(config)
        self.finishing_linear_layer = nn.Linear(config.hidden_size,
                                                config.hidden_size)
        self.dropout1 = nn.Dropout(config.syntax['layer_prepostprocess_dropout'])
        self.ln_2 = nn.LayerNorm(config.hidden_size,
                                 eps=1e-6)
        if config.syntax['tf_enc_use_ffn']:
            self.feed_forward = FeedForwardLayer(config,
                                                 config.syntax['gelu_dropout'])
            self.dropout2 = nn.Dropout(config.syntax['layer_prepostprocess_dropout'])
            self.ln_3 = nn.LayerNorm(config.hidden_size,
                                     eps=1e-6)
        if self.config.syntax['tf_enc_gated_connection']:
            self.gate1 = eval(config.syntax['tf_enc_gate'])(config.hidden_size)
            if config.syntax['tf_enc_use_ffn']:
                self.gate2 = eval(config.syntax['tf_enc_gate'])(config.hidden_size)

    def forward(self, e, attention_mask, dep_rel_matrix=None):
        sub = self.finishing_linear_layer(self.syntax_attention(self.ln_2(e),
                                                                attention_mask,
                                                                dep_rel_matrix)[0])
        sub = self.dropout1(sub)
        if self.config.syntax['tf_enc_act_at_skip_connection']:
            sub = F.gelu(sub)
        if self.config.syntax['tf_enc_gated_connection']:
            e = self.gate1(e, sub)
        else:
            e = e + sub

        if self.config.syntax['tf_enc_use_ffn']:
            sub = self.feed_forward(self.ln_3(e))
            sub = self.dropout2(sub)
            if self.config.syntax['tf_enc_act_at_skip_connection']:
                sub = F.gelu(sub)
            if self.config.syntax['tf_enc_gated_connection']:
                e = self.gate2(e, sub)
            else:
                e = e + sub
        return e


class FeedForwardLayer(nn.Module):
    def __init__(self, config, activation_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.W_1 = nn.Linear(config.hidden_size,
                             config.intermediate_size)
        self.act = modeling_bert.ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(activation_dropout)
        self.W_2 = nn.Linear(config.intermediate_size,
                             config.hidden_size)

    def forward(self, e):
        e = self.dropout(self.act(self.W_1(e)))
        e = self.W_2(e)
        return e


class RelationalBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(RelationalBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size,
                               self.all_head_size)
        self.key = nn.Linear(config.hidden_size,
                             self.all_head_size)
        self.value = nn.Linear(config.hidden_size,
                               self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, dep_rel_matrix=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        rel_attention_scores = 0
        if self.config.syntax['use_dep_rel']:
            rel_attention_scores = query_layer[:, :, :, None, :] * dep_rel_matrix[:, None, :, :, :]
            rel_attention_scores = torch.sum(rel_attention_scores, -1)

        attention_scores = (attention_scores + rel_attention_scores) / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs,
                                     value_layer)

        if self.config.syntax['use_dep_rel']:
            val_edge = attention_probs[:, :, :, :, None] * dep_rel_matrix[:, None, :, :, :]
            context_layer = context_layer + torch.sum(val_edge, -2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
