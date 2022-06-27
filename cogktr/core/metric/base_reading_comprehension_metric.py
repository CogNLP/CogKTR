from cogktr.core.metric.base_metric import BaseMetric
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from argparse import Namespace

class BaseMRCMetric(BaseMetric):
    def __init__(self):
        super(BaseMRCMetric, self).__init__()
        self.start_logits_list = list()
        self.end_logits_list = list()
        self.example_list = list()
        self.feature_list = list()
        self.n_best_size = 20

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
        for (feature_index, feature) in enumerate(self.feature_list):
            start_logits,end_logits = self.start_logits_list[feature_index],self.end_logits_list[feature_index]
            start_indexes = _get_best_indexes(start_logits, self.n_best_size)
            end_indexes = _get_best_indexes(end_logits, self.n_best_size)
            feature_null_score = start_logits[0] + end_logits[0]


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
