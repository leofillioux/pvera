import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRA(nn.Module):
    def __init__(self, linear_layer, in_dim, rank, alpha):
        super(DoRA, self).__init__()
        weight_q, weight_k, weight_v = linear_layer.weight.chunk(3, 0)
        bias_q, bias_k, bias_v = linear_layer.bias.chunk(3, -1)
        self.weight_q = nn.Parameter(weight_q, requires_grad=False)
        self.weight_k = nn.Parameter(weight_k, requires_grad=False)
        self.weight_v = nn.Parameter(weight_v, requires_grad=False)
        self.bias_q = nn.Parameter(bias_q, requires_grad=False)
        self.bias_k = nn.Parameter(bias_k, requires_grad=False)
        self.bias_v = nn.Parameter(bias_v, requires_grad=False)

        self.mag_q = nn.Parameter(self.weight_q.norm(p=2, dim=0, keepdim=True))
        self.mag_k = nn.Parameter(self.weight_k.norm(p=2, dim=0, keepdim=True))
        self.mag_v = nn.Parameter(self.weight_v.norm(p=2, dim=0, keepdim=True))

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())

        self.adapter_A_q = nn.Parameter(torch.randn(in_dim, rank)*std_dev)
        self.adapter_A_v = nn.Parameter(torch.randn(in_dim, rank)*std_dev)
        self.adapter_B_q = nn.Parameter(torch.zeros(rank, in_dim))
        self.adapter_B_v = nn.Parameter(torch.zeros(rank, in_dim))
        self.adapter_alpha = alpha / rank

    def forward(self, x):
        lora_q = torch.matmul(self.adapter_A_q, self.adapter_B_q)
        lora_v = torch.matmul(self.adapter_A_v, self.adapter_B_v)

        adapted_q = self.weight_q + lora_q
        column_norm_q = adapted_q.norm(p=2, dim=0, keepdim=True)
        norm_adapted_q = adapted_q / column_norm_q
        weight_q = self.mag_q * norm_adapted_q

        adapted_k = self.weight_k
        column_norm_k = adapted_k.norm(p=2, dim=0, keepdim=True)
        norm_adapted_k = adapted_k / column_norm_k
        weight_k = self.mag_k * norm_adapted_k

        adapted_v = self.weight_v + lora_v
        column_norm_v = adapted_v.norm(p=2, dim=0, keepdim=True)
        norm_adapted_v = adapted_v / column_norm_v
        weight_v = self.mag_v * norm_adapted_v

        x_q = F.linear(x, weight_q, self.bias_q)
        x_k = F.linear(x, weight_k, self.bias_k)
        x_v = F.linear(x, weight_v, self.bias_v)

        x_adapted = torch.cat([x_q, x_k, x_v], dim=-1)
        return x_adapted

class GridSearchDoRA(nn.Module):
    def __init__(self, nb_heads, linear_layer, in_dim, rank, alpha):
        super(GridSearchDoRA, self).__init__()
        self.nb_heads = nb_heads
        if nb_heads == 1:
            self.heads = DoRA(linear_layer, in_dim, rank[0], alpha)
        else:
            self.heads = nn.ModuleList([DoRA(linear_layer, in_dim, rank[i], alpha)
                                         for i in range(nb_heads)])

    def forward(self, x):
        if isinstance(self.heads, DoRA):
            return self.heads(x)
        else:
            bs = x.shape[0] // self.nb_heads
            assert x.shape[0] % self.nb_heads == 0
            x = [self.heads[i](x[i * bs : (i + 1) * bs]) for i in range(self.nb_heads)]
            return torch.cat(x, dim=0)