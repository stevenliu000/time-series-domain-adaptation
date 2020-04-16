
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class BinaryLossNewImplementation(nn.Module):
    def __init__(self, device):
        super(BinaryLossNewImplementation, self).__init__()
        self.device = device    

    def forward(self, fake_x_embedding, source_x_embedding, num_class, num_per_class, mask=None):
        """
        fake_x_embedding:  [x1, x1, x1, ..., x2, x2, x2, ..., xn, xn, xn, ...]
        source_x_embedding: [x1, x1, x1, ..., x2, x2, x2, ..., xn, xn, xn, ...]
        (where n is num_class)
        
        for each class, data repeats num_per_class times.
        """
        if type(mask) == torch.Tensor:
            mask = mask.to(self.device)
        elif type(mask) == np.ndarray:
            mask = torch.LongTensor(mask).to(device)
        else:
            mask = torch.ones([num_per_class * num_class, num_per_class * num_class]).to(self.device)
        assert mask.shape == torch.Size([num_per_class * num_class, num_per_class * num_class])
        
        labels = torch.zeros([num_per_class * num_class, num_per_class * num_class])
        for i in range(num_class):
            labels[i*num_per_class:(i+1)*num_per_class, i*num_per_class:(i+1)*num_per_class] = 1
        labels = labels.to(self.device) # labels is a 1 block-diagnoal matrix
        ones_l1_norm = torch.sum(labels) # number of pairs whose labels are 1
        zeros_l1_norm = torch.sum(1-labels) # number of pairs whose labels are 0
        
        ones_weights = labels / ones_l1_norm
        zeros_weights = (1 - labels) / zeros_l1_norm
        weights = ones_weights + zeros_weights
        weights = weights.to(self.device)
        weights = weights * mask
        
        logits = torch.matmul(fake_x_embedding, source_x_embedding.transpose(0,1))
        # weights already includes the normalization term
        loss_mean = F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction='sum')

        return loss_mean

