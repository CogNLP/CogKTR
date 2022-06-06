from cogktr.models.base_model import BaseModel
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.hidden_1 = nn.Linear(input_dim, hidden_dim)
        self.act_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        x=self.hidden_1(x)
        x=self.act_1(x)
        x=self.hidden_2(x)
        return x



class KgembModel4TC(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.bert = BertModel.from_pretrained(plm)
        self.word_embedding=self.bert.get_input_embeddings().weight

        # TODO: auto select input size
        self.input_size = 768
        self.classes_num = len(vocab["label_vocab"])
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)
        self.mlp=MLP(input_dim=100,hidden_dim=1000,output_dim=self.input_size)

    def loss(self, batch, loss_function):
        batch_len = len(batch)
        token,label,entity_mask_list,entity_embedding_list=self.get_batch(batch=batch)
        pred = self.forward(token=token, entity_mask_list=entity_mask_list,entity_embedding_list=entity_embedding_list)
        loss = loss_function(pred, label) / batch_len
        return loss

    def forward(self, token, entity_mask_list,entity_embedding_list):
        word_embeds=self.word_embedding[token]
        entity_embeds=[]
        for entity_masks,entity_embeddings in zip(entity_mask_list,entity_embedding_list):
            flag=0
            for entity_mask,entity_embedding in zip(entity_masks,entity_embeddings):
                if flag==1:
                    entity_embed+=torch.FloatTensor(entity_mask).to("cuda")*self.mlp(torch.FloatTensor(entity_embedding).squeeze().to("cuda"))
                if flag==0:
                    entity_embed=torch.FloatTensor(entity_mask).to("cuda")*self.mlp(torch.FloatTensor(entity_embedding).squeeze().to("cuda"))
                    flag=1
            entity_embeds.append(entity_embed)

        entity_embeds=torch.stack(entity_embeds)
        inputs_embeds=word_embeds+entity_embeds
        x = self.bert(inputs_embeds=inputs_embeds).pooler_output
        ######方式一
        # x = self.bert(token).pooler_output
        ######方式二
        # inputs_embeds=self.word_embedding[token]
        # x = self.bert(inputs_embeds=inputs_embeds).pooler_output
        ###### 以上两种方式效果一样
        x = self.linear(x)
        return x

    def evaluate(self, batch, metric_function):
        token,label,entity_mask_list,entity_embedding_list=self.get_batch(batch=batch)
        pred = self.predict(token,entity_mask_list=entity_mask_list,entity_embedding_list=entity_embedding_list)
        metric_function.evaluate(pred, label)

    def predict(self, token,entity_mask_list,entity_embedding_list):
        pred = self.forward(token=token, entity_mask_list=entity_mask_list,entity_embedding_list=entity_embedding_list)
        pred = F.softmax(pred, dim=1)
        pred = torch.max(pred, dim=1)[1]
        return pred

    def get_batch(self,batch):
        token = torch.stack(batch["token"])
        label = torch.stack(batch["label"])
        entity_mask_list=batch["entity_mask_list"]
        entity_embedding_list=batch['entity_embedding_list']
        return token,label,entity_mask_list,entity_embedding_list

