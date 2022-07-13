from cogktr.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import top_k_accuracy_score
import numpy as np


class BaseQuestionAnsweringMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.label_list = list()
        self.pre_list = list()

    def evaluate(self, pred, label):
        # print(pred.size())
        # print(label.size())
        self.label_list = self.label_list + label.cpu().tolist()
        self.pre_list = self.pre_list + pred.cpu().tolist()

    def get_metric(self, reset=True, topk=5):
        # print(self.pre_list.size())
        # print(self.label_list.size())
        # print(max(self.label_list))
        # print(len(self.pre_list[0]))
        acc = top_k_accuracy_score(self.label_list, self.pre_list, k=topk, labels=range(len(self.pre_list[0])))
        evaluate_result = {"Acc": acc}
        if reset:
            self.label_list = list()
            self.pre_list = list()
        return evaluate_result