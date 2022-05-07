from cogktr.core.metric.basemetric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class BaseTextClassificationMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.pre_list = list()
        self.label_list = list()

    def evaluate(self,pred, label):
        self.pre_list=self.pre_list+pred.cpu().tolist()
        self.label_list=self.label_list+label.cpu().tolist()

    def get_metric(self,reset=True):
        micro_P = precision_score(self.pre_list, self.label_list, average="micro")
        micro_R = recall_score(self.pre_list, self.label_list, average="micro")
        micro_F1 = f1_score(self.pre_list, self.label_list, average="micro")
        macro_P = precision_score(self.pre_list, self.label_list, average="micro")
        macro_R = recall_score(self.pre_list, self.label_list, average="micro")
        macro_F1 = f1_score(self.pre_list, self.label_list, average="micro")
        evaluate_result={"micro_P":micro_P,
                         "micro_R":micro_R,
                         "micro_F1":micro_F1,
                         "macro_P": macro_P,
                         "macro_R": macro_R,
                         "macro_F1": macro_F1,
                         }
        if reset:
            self.pre_list = list()
            self.label_list = list()
        return evaluate_result
