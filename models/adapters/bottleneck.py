import torch
import torch.nn as nn

class BNAdapter(nn.Module):
    def __init__(self, in_dim, adapter_activation, reduction_factor):
        super(BNAdapter, self).__init__()
        hidden_dim = int(in_dim/reduction_factor)
        self.adapter_activation = getattr(nn, adapter_activation)()
        self.adapter_downsample = nn.Linear(in_dim, hidden_dim)
        self.adapter_upsample = nn.Linear(hidden_dim, in_dim)

    def forward(self, x_in):
        x = self.adapter_downsample(x_in)
        x = self.adapter_activation(x)
        x = self.adapter_upsample(x)
        x += x_in
        return x

class GridSearchBNAdapter(nn.Module):
    def __init__(self, nb_heads, in_dim, adapter_activation, reduction_factor):
        super(GridSearchBNAdapter, self).__init__()
        self.nb_heads = nb_heads
        if nb_heads == 1:
            self.heads = BNAdapter(in_dim, adapter_activation, reduction_factor[0])
        else:
            self.heads = nn.ModuleList([BNAdapter(in_dim, adapter_activation, reduction_factor[i])
                                         for i in range(nb_heads)])

    def forward(self, x):
        if isinstance(self.heads, BNAdapter):
            return self.heads(x)
        else:
            bs = x.shape[0] // self.nb_heads
            assert x.shape[0] % self.nb_heads == 0
            x = [self.heads[i](x[i * bs : (i + 1) * bs]) for i in range(self.nb_heads)]
            return torch.cat(x, dim=0)