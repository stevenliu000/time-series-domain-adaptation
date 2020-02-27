import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modules.transformer import TransformerEncoder
from modules.complex_unit import ComplexLinear
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ComplexTransformer(nn.Module):
    def __init__(self, layers, time_step, input_dim, hidden_size, output_dim, num_heads, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, out_dropout=0.5, attn_mask=False):
        super(ComplexTransformer, self).__init__()
        self.orig_d_a = self.orig_d_b = input_dim
        self.d_a, self.d_b = 512, 512 
        h_out = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask

        # because this is an encoder which preserves shape, embed_dim=input_dim
        self.embed_dim = input_dim
        
        # Transformer networks
        self.trans = self.get_network()
        self.fc_a = nn.Linear(self.orig_d_a, self.d_a)
        self.fc_b = nn.Linear(self.orig_d_b, self.d_b)
        # Projection layers
        self.proj = ComplexLinear(self.d_a, self.embed_dim)
    def get_network(self):
        
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers, attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout, res_dropout=self.res_dropout, attn_mask=self.attn_mask)
            
    def forward(self, input_a, input_b):
        """
        x_a and x_b should have dimension [batch_size, seq_len, n_features] (i.e., N, L, C).
        """
        input_a = input_a.transpose(0, 1)
        input_b = input_b.transpose(0, 1)
        input_a = self.fc_a(input_a)
        input_b = self.fc_b(input_b)
        input_a, input_b = self.proj(input_a, input_b)
        h_as, h_bs = self.trans(input_a, input_b)
        h_as = h_as.transpose(0, 1)
        h_bs = h_bs.transpose(0, 1)
        return h_as, h_bs

