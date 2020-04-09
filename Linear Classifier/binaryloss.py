#!/usr/bin/env python
# coding: utf-8

# In[26]:


import torch.nn as nn
import torch



class BinaryLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.BCE = nn.BCELoss()
    

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
        logit_all = torch.sum(target_x_embedding_rand * sourece_x_embedding_rand, axis=1)
        # same class label 1, else 0
        class_same = (target_y_rand == source_y_rand).float()
        output = self.BCE(self.sigmoid(logit_all), class_same)
        return output


# In[27]:


# target_x_embedding = torch.randn((100,500), requires_grad=True)
# source_x_embedding = torch.randn((100,500), requires_grad=True)

# target_y = torch.ones(100,)
# source_y = torch.ones(100,)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# m = BinaryLoss(device).to(device)
# loss = m(target_x_embedding, target_y, source_x_embedding, source_y)
# loss.backward()
# loss

