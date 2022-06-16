from cogktr.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class BaseMRCMetric(BaseMetric):
    def __init__(self, ):
        super().__init__()
        self.labels_first = list()
        self.labels_second = list()
        self.preds_first = list()
        self.preds_second = list()

    def evaluate(self,preds_first,preds_second,labels_first,labels_second):
        self.labels_first = self.labels_first + labels_first.cpu().tolist()
        self.labels_second = self.labels_second + labels_second.cpu().tolist()
        self.preds_first = self.preds_first + preds_first.cpu().tolist()
        self.preds_second = self.preds_second + preds_second.cpu().tolist()

    def get_metric(self, reset=True):
        total = 0
        match = 0
        for (label_first,label_second,pred_first,pred_second) in zip(
            self.labels_first,self.labels_second,self.preds_first,self.preds_second
        ):
            # print("Label:({},{})  Predict:({},{})".format(
            #     label_first,label_second,pred_first,pred_second
            # ))
            total += 1
            if label_first == pred_first and label_second == pred_second:
                match += 1

        ExactMatch = match / total

        evaluate_result = {
            "EM":ExactMatch,
        }
        if reset:
            self.labels_first = list()
            self.labels_second = list()
            self.preds_first = list()
            self.preds_second = list()
        return evaluate_result
