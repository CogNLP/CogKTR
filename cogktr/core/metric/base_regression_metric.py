from cogktr.core.metric.base_metric import BaseMetric
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


class BaseRegressionMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.pre_list = list()
        self.label_list = list()

    def evaluate(self, pred, label):
        self.pre_list = self.pre_list + pred.cpu().tolist()
        self.label_list = self.label_list + label.cpu().tolist()

    def get_metric(self, reset=True):
        evaluate_result = {}
        # TODO:Check whether R2 is negative is correct
        r2 = r2_score(self.pre_list, self.label_list)
        mse = mean_squared_error(self.pre_list, self.label_list)
        mae = mean_absolute_error(self.pre_list, self.label_list)
        pear=pearsonr(self.pre_list,self.label_list)[0]
        evaluate_result = {"r2": r2,
                           "mse": mse,
                           "mae": mae,
                           "pear":pear,
                           }
        if reset:
            self.pre_list = list()
            self.label_list = list()
        return evaluate_result
