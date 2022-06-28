from cogktr.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from argparse import Namespace
import collections
from collections import defaultdict,namedtuple
from cogktr.utils.log_utils import logger
import re
import string
import numpy as np

class BaseMRCMetric(BaseMetric):
    def __init__(self):
        super(BaseMRCMetric, self).__init__()
        self.start_logits_list = list()
        self.end_logits_list = list()
        self.example_list = list()
        self.feature_list = list()
        self.n_best_size = 20
        self.max_answer_length = 30

    def evaluate(self,start_logits,end_logits,batch):
        self.start_logits_list = self.start_logits_list + start_logits.cpu().numpy().tolist()
        self.end_logits_list = self.end_logits_list + start_logits.cpu().numpy().tolist()
        feature_list = []
        for i in range(len(batch["example"])):
            feature1 = {
                key:value[i].cpu().tolist() for key,value in batch.items() if key != "example" and key != "additional_info"
            }
            feature2 = vars(batch["additional_info"][i])
            feature_list.append(Namespace(**feature1,**feature2))
        self.feature_list = self.feature_list + feature_list
        self.example_list = self.example_list + batch["example"]

    def get_metric(self, reset=True):
        print("!")
        qas_id2features = defaultdict(list)
        qas_id2example = defaultdict(list)
        qas_id2em_score = dict()
        qas_id2f1_score = dict()
        for index,feature in enumerate(self.feature_list):
            qas_id2features[self.example_list[index].qas_id].append(self.feature_list[index])
            qas_id2example[self.example_list[index].qas_id] = self.example_list[index]

        Prediction =collections.namedtuple(
            "prediction",["text","start_logit","end_logit"]
        )
        for index,(qas_id,example) in enumerate(qas_id2example.items()):
            features = qas_id2features[qas_id]
            score_null = 1000000
            null_start_logit = 0
            null_end_logit = 0
            seen_predictions = {}
            n_best  =[]
            for (feature_index, feature) in enumerate(features):
                start_logits,end_logits = self.start_logits_list[feature_index],self.end_logits_list[feature_index]
                start_indexes = _get_best_indexes(start_logits, self.n_best_size)
                end_indexes = _get_best_indexes(end_logits, self.n_best_size)
                feature_null_score = start_logits[0] + end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    null_start_logit = start_logits[0]
                    null_end_logit = end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            continue
                        # after obtain one valid span:
                        tok_tokens = feature.tokens[start_index:(end_index + 1)]
                        tok_text = " ".join(tok_tokens)
                        tok_text = tok_text.replace(" ##", "")
                        tok_text = tok_text.replace("##", "")
                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())
                        final_text = tok_text
                        if final_text in seen_predictions:
                            continue
                        seen_predictions[final_text] = True
                        n_best.append(
                            Prediction(
                                text=final_text,
                                start_logit=start_logits[start_index],
                                end_logit=end_logits[end_index]
                            )
                        )

            n_best.append(Prediction(
                text = "",
                start_logit=null_start_logit,
                end_logit=null_end_logit,
            ))
            n_best = sorted(
                n_best,
                key=lambda x:(x.start_logit+x.end_logit),
                reverse=True, # Descending Order
            )
            n_best = n_best if len(n_best) < self.n_best_size else n_best[:self.n_best_size]
            pred_text = n_best[0].text
            gold_text = example.answer_text
            em_score = compute_exact(gold_text,pred_text)
            f1_score = compute_f1(gold_text,pred_text)
            qas_id2em_score[qas_id] = em_score
            qas_id2f1_score[qas_id] = f1_score

        eval_result = {
            "EM":np.mean(np.array([list(qas_id2em_score.values())])),
            "F1":np.mean(np.array([list(qas_id2f1_score.values())])),
        }
        if reset:
            self.start_logits_list = list()
            self.end_logits_list = list()
            self.example_list = list()
            self.feature_list = list()
        return eval_result







def normalize_answer(s):

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1









def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

# class BaseMRCMetric(BaseMetric):
#     def __init__(self, ):
#         super().__init__()
#         self.labels_first = list()
#         self.labels_second = list()
#         self.preds_first = list()
#         self.preds_second = list()
#
#     def evaluate(self,preds_first,preds_second,labels_first,labels_second):
#         self.labels_first = self.labels_first + labels_first.cpu().tolist()
#         self.labels_second = self.labels_second + labels_second.cpu().tolist()
#         self.preds_first = self.preds_first + preds_first.cpu().tolist()
#         self.preds_second = self.preds_second + preds_second.cpu().tolist()
#
#     def get_metric(self, reset=True):
#         total = 0
#         match = 0
#         for (label_first,label_second,pred_first,pred_second) in zip(
#             self.labels_first,self.labels_second,self.preds_first,self.preds_second
#         ):
#             # print("Label:({},{})  Predict:({},{})".format(
#             #     label_first,label_second,pred_first,pred_second
#             # ))
#             total += 1
#             if label_first == pred_first and label_second == pred_second:
#                 match += 1
#
#         ExactMatch = match / total
#
#         evaluate_result = {
#             "EM":ExactMatch,
#         }
#         if reset:
#             self.labels_first = list()
#             self.labels_second = list()
#             self.preds_first = list()
#             self.preds_second = list()
#         return evaluate_result
