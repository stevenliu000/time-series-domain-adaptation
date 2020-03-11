import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
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

            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            #nn.Tanh()
        ) 

    def forward(self, x):
        # x: [bs, seq, init_size (small)]
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
        self.fc = nn.Linear(3200, self.d_out)
    def forward(self, x, mask):
        # x: [bs, seq, feature_dim]
        batch_size, seq_len, feature_dim = x.shape
        x = self.net(x)
        x = x.reshape(batch_size, -1)
        out = self.fc(x)
        if self.d_out != 1:
            out = out * mask
        return out