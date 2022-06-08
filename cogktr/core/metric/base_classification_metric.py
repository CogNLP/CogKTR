from cogktr.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class BaseClassificationMetric(BaseMetric):
    def __init__(self, mode):
        super().__init__()
        if mode not in ["binary", "multi"]:
            raise ValueError("Please choose mode in binary or multi")
        self.mode = mode
        self.label_list = list()
        self.pre_list = list()

    def evaluate(self, pred, label):
        self.label_list = self.label_list + pred.cpu().tolist()
        self.pre_list = self.pre_list + label.cpu().tolist()

    def get_metric(self, reset=True):
        evaluate_result = {}
        if self.mode == "binary":
            P = precision_score(self.label_list, self.pre_list, average="binary")
            R = recall_score(self.label_list, self.pre_list, average="binary")
            F1 = f1_score(self.label_list, self.pre_list, average="binary")
            evaluate_result = {"P": P,
                               "R": R,
                               "F1": F1,
                               }
        if self.mode == "multi":
            micro_P = precision_score(self.label_list, self.pre_list, average="micro")
            micro_R = recall_score(self.label_list, self.pre_list, average="micro")
            micro_F1 = f1_score(self.label_list, self.pre_list, average="micro")
            macro_P = precision_score(self.label_list, self.pre_list, average="macro")
            macro_R = recall_score(self.label_list, self.pre_list, average="macro")
            macro_F1 = f1_score(self.label_list, self.pre_list, average="macro")
            evaluate_result = {"micro_P": micro_P,
                               "micro_R": micro_R,
                               "micro_F1": micro_F1,
                               "macro_P": macro_P,
                               "macro_R": macro_R,
                               "macro_F1": macro_F1,
                               }
        if reset:
            self.label_list = list()
            self.pre_list = list()
        return evaluate_result
