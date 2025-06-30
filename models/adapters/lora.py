import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, linear_layer, in_dim, rank, alpha):
        super(LoRA, self).__init__()
        self.linear_layer = linear_layer
        std = 1 / torch.sqrt(torch.tensor(rank).float())
        self.adapter_Q_downsample = nn.Parameter(torch.randn(in_dim, rank) * std)
        self.adapter_Q_upsample = nn.Parameter(torch.zeros(rank, in_dim))
        self.adapter_V_downsample = nn.Parameter(torch.randn(in_dim, rank) * std)
        self.adapter_V_upsample = nn.Parameter(torch.zeros(rank, in_dim))
        self.adapter_alpha = alpha / rank

    def forward(self, x):
        x_q = self.adapter_alpha * (x @ self.adapter_Q_downsample @ self.adapter_Q_upsample)
        x_v = self.adapter_alpha * (x @ self.adapter_V_downsample @ self.adapter_V_upsample)
        x_lora = torch.cat([x_q, torch.zeros_like(x_v), x_v], dim=-1)
        x = self.linear_layer(x) + x_lora
        return x

class GridSearchLoRA(nn.Module):
    def __init__(self, nb_heads, linear_layer, in_dim, ranks, alpha):
        super(GridSearchLoRA, self).__init__()
        self.nb_heads = nb_heads
        if nb_heads == 1:
            self.heads = LoRA(linear_layer, in_dim, ranks[0], alpha)
        else:
            self.heads = nn.ModuleList([LoRA(linear_layer, in_dim, ranks[i], alpha) for i in range(nb_heads)])

    def forward(self, x):
        if isinstance(self.heads, LoRA):
            x = self.heads(x)
        else:
            bs = x.shape[0] // self.nb_heads
            assert x.shape[0] % self.nb_heads == 0
            x = [self.heads[i](x[i * bs : (i + 1) * bs])
                 for i in range(self.nb_heads)]
            x = torch.concatenate(x, dim=0)
        return x
