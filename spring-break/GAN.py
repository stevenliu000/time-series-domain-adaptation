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
        self.fc = nn.Linear(3200, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: [bs, seq, feature_dim]
        x = self.net(x)
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        out = self.sigmoid(self.fc(x))
        return out
