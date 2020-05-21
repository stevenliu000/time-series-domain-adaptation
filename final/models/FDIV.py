import torch
import torch.nn as nn

class Gfunction(nn.Sequential):
    def __init__(self, num_class):
        super(Gfunction, self).__init__(
            nn.Linear(128+num_class,100),
            nn.ELU(),
            nn.Linear(100,100),
            nn.ELU(),
            nn.Linear(100, num_class)
        )


def JSDiv(g_x_source, g_x_target, mask_source, mask_target, device):
    # class one hot encoding mask 
    s_score = F.softplus(-g_x_source) * mask_source # (batch_size, num_class)
    t_score = F.softplus(g_x_target) * mask_target # (batch_size, num_class)

    # E_{p(x|y)}
    s_score = s_score.sum(dim=0) / mask_source.sum(dim=0) # (num_class, )
    t_score = t_score.sum(dim=0) / mask_target.sum(dim=0) # (num_class, )

    # E_{p(y)}
    return (- s_score - t_score).mean() # scalar