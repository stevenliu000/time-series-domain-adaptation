#!/usr/bin/env python
# coding: utf-8

# In[58]:


import torch



class BinaryLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    

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
        logit_all = target_x_embedding_rand * sourece_x_embedding_rand
        
        # same class label 1, else 0
        class_same = (target_y_rand == source_y_rand).long()
        
        criterion_crossentropy = torch.nn.CrossEntropyLoss()
        loss = criterion_crossentropy(logit_all, class_same)
        return loss


# In[59]:


# a = torch.randn((2,3), requires_grad=True)
# b = torch.randn((2,3), requires_grad=True)

# a_lbl = torch.randint(0,10,(2,))
# b_lbl = torch.randint(0,10,(2,))
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# m = BinaryLoss().to(device)
# loss = m(a, a_lbl, b, b_lbl)
# loss.backward()
# loss

