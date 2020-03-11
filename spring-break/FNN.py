import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, d_in, d_h, d_out, dp=0.2):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.ln = nn.LayerNorm(d_h)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dp)
        self.fc2 = nn.Linear(d_h, d_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x