import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(300, 500),
            nn.LayerNorm(500),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(500, 500),
            nn.LayerNorm(500),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(500, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(300, dim),
            nn.LayerNorm(dim),
        ) 

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, feature_dim, d_out):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ) 

        self.d_out = d_out
        self.fc = nn.Linear(feature_dim, self.d_out)
    def forward(self, x, mask):
        x = self.net(x)
        out = self.fc(x)
        if self.d_out != 1:
            out = out * mask
        return out