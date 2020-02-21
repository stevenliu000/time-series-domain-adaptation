from torch import nn

class Classifier(nn.Module):
    def __init__(self, layers_size, dim_out, dropout=0.3):
        super(Classifier, self).__init__()
        self.layers = []
        self.activs = []
        self.dropouts = []
        self.layers_size = layers_size
        
        for i in range(len(layers_size)-1):
            self.layers.append(nn.Linear(layers_size[i], layers_size[i+1]))
            self.activs.append(nn.ReLU())
            if i < len(layers_size)-2:
                self.dropouts.append(nn.Dropout(p=dropout))

        self.layers.append(nn.Linear(layers_size[-1], dim_out))
        self.nlayer = len(self.layers)
        self.nactivs = len(self.activs)
        self.ndropouts = len(self.dropouts)

        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        out = x
        for i in range(self.nlayer):
            out = self.layers[i](out)
            if i < self.nactivs:
                out = self.activs[i](out)
            if i < self.ndropouts:
                out = self.dropouts[i](out)

        return out
