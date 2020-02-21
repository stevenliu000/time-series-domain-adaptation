from torch import nn 

class Discriminator(nn.Module):
    '''
    credit: from https://github.com/martinmamql/complex_da
    '''
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

class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        self.transformer_layer_args = {'d_model':2, 'nhead':2, 'dim_feedforward':1024, 'dropout':0.1, 'activation':'gelu'}
        self.transformer_args = {'num_layers':3, 'norm':None}
        self.transformer_layer_args.update(kwargs['transformer_layer'])
        self.transformer_args.update(kwargs['transformer'])
        
        self.transformer_layer = nn.TransformerEncoderLayer(**self.transformer_layer_args)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, **self.transformer_args)
        
    def forward(self, x):
        out = self.transformer(x)
        return out
        