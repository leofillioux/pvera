import torch
import torch.nn as nn
from einops import repeat

class IA3Attention(nn.Module):
    def __init__(self, linear_layer, in_dim):
        super(IA3Attention, self).__init__()
        self.linear_layer = linear_layer
        self.adapter_lv = nn.Parameter(torch.ones(in_dim))
        self.adapter_lk = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        x = self.linear_layer(x)
        x_q, x_k, x_v = x.chunk(3, dim=-1)

        x_v = x_v * repeat(self.adapter_lv, 'd -> b l d', b=x_v.shape[0], l=x_v.shape[1])
        x_k = x_k * repeat(self.adapter_lk, 'd -> b l d', b=x_k.shape[0], l=x_k.shape[1])
        x = torch.cat([x_q, x_k, x_v], dim=-1)
        return x

class IA3Linear(nn.Module):
    def __init__(self, activation, in_dim):
        super(IA3Linear, self).__init__()
        self.activation = activation
        self.adapter_lff = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        x = self.activation(x)
        x = x * repeat(self.adapter_lff, 'd -> b l d', b=x.shape[0], l=x.shape[1])
        return x
