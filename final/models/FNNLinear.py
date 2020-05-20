import torch
import torch.nn as nn

class FNNLinear(nn.Module):
    def __init__(self, d_h2, d_out):
        super(FNNLinear, self).__init__()
        self.fc3 = nn.Linear(d_h2, d_out)

    def forward(self, x):
        x = self.fc3(x)

        return x
