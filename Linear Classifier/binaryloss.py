#!/usr/bin/env python
# coding: utf-8

# In[128]:


import torch.nn.functional as F
import torch
from torch.autograd import Variable


class BinaryLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    

    def forward(self, target_x_embedding, target_y, source_x_embedding, source_y):
        """
        target_x_embedding:  torch.Size([100, 500])
        target_y: torch.Size([100])
        source_x_embedding: torch.Size([100, 500])
        source_y: torch.Size([100])
        """
        # combination
        
        assert target_x_embedding.size(0) == source_x_embedding.size(0)
        target_index = torch.randperm(target_x_embedding.size(0))
        source_index = torch.randperm(source_x_embedding.size(0))
        target_x_embedding_rand = target_x_embedding[target_index]
        sourece_x_embedding_rand = source_x_embedding[source_index]
        target_y_rand = target_y[target_index]
        source_y_rand = source_y[source_index]
        
        # logit as vector innner product
        logit_all = torch.sigmoid(torch.sum(target_x_embedding_rand * sourece_x_embedding_rand, axis=1)).view(-1,1)
        
        input_ = torch.cat((1 - logit_all, logit_all), axis=1)
        
        # same class label 1, else 0
        class_same = (target_y_rand == source_y_rand).long().view(-1,1)
        class_same_one_hot = torch.FloatTensor(target_x_embedding.size(0), 2).to(self.device)
        class_same_one_hot.zero_()
        class_same_one_hot.scatter_(1, class_same, 1)

        loss = F.binary_cross_entropy(input_, class_same_one_hot)

        return loss


# In[132]:


# a = torch.ones((2,3), requires_grad=True)
# b = torch.ones((2,3), requires_grad=True)

# a_lbl = torch.randint(0,1,(2,))
# b_lbl = torch.randint(0,1,(2,))
# print(a_lbl)
# print(b_lbl)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# m = BinaryLoss().to(device)
# loss = m(a, a_lbl, b, b_lbl)
# loss.backward()
# loss


# In[ ]:




