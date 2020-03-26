#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

class PrototypeLoss(nn.Module):
    """
    Prototype loss.

    Reference: Jake Snell et al. Prototypical Networks for Few-shot Learning. https://arxiv.org/pdf/1703.05175.pdf

    """
    def __init__(self, use_gpu, device):
        super(PrototypeLoss, self).__init__()
        self.use_gpu = use_gpu
        self.device = device



    def forward(self, pred, target_y, centers):
        """
        note that we randomly select the
        Args
        pred: source x from a batch: (batch_size, feature_dim)
        target_y: source x from a batch: (batch_size, )
        centers: (class_num, feature_dim)
        """

        # get source data for all class y
#         source_dataloader_iter = iter(source_dataloader)
#         source_proto_x, source_proto_y = next(source_dataloader_iter)
#         source_proto_x = source_proto_x.to(device)
#         source_proto_y = source_proto_y.to(device)
#         class_fetched = torch.unique(source_proto_y)
#         source_data_fetched_p, source_y_fetched_p = source_proto_x, source_proto_y
#         class_list = torch.unique(target_y)
#         print(source_proto_x.shape)
#         while torch.any(class_fetched != class_list):
#             source_proto_x, source_proto_y = next(source_dataloader_iter)
#             print(source_proto_x.shape)
#             source_proto_x = source_proto_x.to(device)
#             source_proto_y = source_proto_y.to(device)
#             source_data_fetched_p = torch.cat(source_data_fetched_p, source_proto_x)
#             source_y_fetched_p = torch.cat(source_y_fetched_p, source_proto_y)
#             new_class_fetched = torch.unique(source_proto_y)
#             class_fetched = torch.unique(torch.cat((class_fetched, new_class_fetched)))


        # compute mean of each class in class_list
        # center_source = self.compute_mean(source_data_fetched_p, source_y_fetched_p)
        def compute_mean(samples, labels):
            assert samples.size(0) == labels.size(0)
            """
            samples = torch.Tensor([
                                 [0.1, 0.1],    #-> group / class 1
                                 [0.2, 0.2],    #-> group / class 2
                                 [0.4, 0.4],    #-> group / class 2
                                 [0.0, 0.0]     #-> group / class 0
                          ])
            labels = torch.LongTensor([1, 2, 2, 0])

            return
                tensor([[0.0000, 0.0000],
                        [0.1000, 0.1000],
                        [0.3000, 0.3000]])
            """
            labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
            unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
            res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
            res = res / labels_count.float().unsqueeze(1)
            return res

        center_batch = centers[target_y, ]
        # compute loss for target data
        ## center_source (K, feature_dim)
        ## pred (batch_size, feature_dim)
        ## target_y (batch_size, )
        # correspond_center = center_source[target_y, ]
        dist = torch.sum(torch.pow(pred - center_batch, 2), axis=1)
        loss = torch.sum(compute_mean(dist.view(dist.shape[0],-1), target_y), axis=0)
        return loss




