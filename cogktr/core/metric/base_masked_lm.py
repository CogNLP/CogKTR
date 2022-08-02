from cogktr.core.metric.base_metric import BaseMetric
from sklearn.metrics import top_k_accuracy_score


class BaseMaskedLMMetric(BaseMetric):
    def __init__(self, topk=1):
        super().__init__()
        self.topk = topk
        self.label_list = list()
        self.pre_list = list()
        self.default_metric_name = "Top_K_Acc"

    def evaluate(self, pred, label):
        self.label_list = self.label_list + label.cpu().tolist()
        self.pre_list = self.pre_list + pred.cpu().tolist()

    def get_metric(self, reset=True):
        top_k_acc = top_k_accuracy_score(self.label_list,
                                         self.pre_list,
                                         k=self.topk,
                                         labels=list(range(len(self.pre_list[0]))))
        evaluate_result = {"Top_K_Acc": top_k_acc}
        if reset:
            self.label_list = list()
            self.pre_list = list()
        return evaluate_result
