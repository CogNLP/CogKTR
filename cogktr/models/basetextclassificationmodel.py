from cogktr.models.basemodel import BaseModel
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch

class BaseTextClassificationModel(BaseModel):
    def __init__(self,vocab,plm):
        super().__init__()
        self.vocab=vocab
        self.plm=plm

        self.bert =BertModel.from_pretrained(plm)
        # TODO: auto select input size
        self.input_size = 768
        self.classes_num = len(vocab["label_vocab"])
        self.linear=nn.Linear(in_features=self.input_size,out_features=self.classes_num)

    def loss(self,batch,loss_function):
        batch_len=len(batch)
        token,label=batch
        pred=self.forward(token=token)
        loss=loss_function(pred,label)/batch_len
        return loss

    def forward(self,token):
        x=self.bert(token).pooler_output
        x=self.linear(x)
        return x

    def evaluate(self,batch,metric_function):
        token, label = batch
        pred=self.predict(token)
        metric_function.evaluate(pred,label)

    def predict(self,token):
        pred=self.forward(token)
        pred= F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred
