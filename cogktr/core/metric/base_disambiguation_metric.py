from cogktr.core.metric.base_metric import BaseMetric
import numpy as np

class BaseDisambiguationMetric(BaseMetric):
    def __init__(self, segment_list):
        super().__init__()
        self.segment_list = segment_list
        self.label_list = list()
        self.pre_list = list()

    def evaluate(self, pred, label):
        self.label_list = self.label_list + label.cpu().tolist()
        self.pre_list = self.pre_list + pred.cpu().tolist()

    def get_metric(self, reset=True):
        label_list=np.array(self.label_list)
        pre_list=np.array(self.pre_list)
        evaluate_result = {}
        ok = 0
        for begin, end in zip(self.segment_list[:-1], self.segment_list[1:]):
            if pre_list[begin:end].argmax() == int(np.where(label_list[begin:end]==1)[0]):
                ok += 1
        F1 =ok / (len(self.segment_list) - 1) * 100
        evaluate_result = {"F1": F1}
        if reset:
            self.label_list = list()
            self.pre_list = list()
        return evaluate_result
