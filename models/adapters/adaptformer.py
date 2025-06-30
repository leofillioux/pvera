import torch
import torch.nn as nn

class AdaptFormer(nn.Module):
    def __init__(self, layer_norm, linear, adapter_activation, reduction_ratio):
        super(AdaptFormer, self).__init__()
        self.layer_norm = layer_norm
        self.linear = linear
        self.adapter_alpha = nn.Parameter(torch.ones(1))

        hidden_dim = int(self.linear.fc1.in_features/reduction_ratio)
        self.adapter_downsample = nn.Linear(self.linear.fc1.in_features, hidden_dim)
        self.adapter_activation = getattr(nn, adapter_activation)()
        self.adapter_upsample = nn.Linear(hidden_dim, self.linear.fc1.in_features)

    def forward(self, x):
        main_x = self.linear(self.layer_norm(x))
        adapted_x = self.adapter_upsample(self.adapter_activation(self.adapter_downsample(x))) * self.adapter_alpha
        return main_x + adapted_x

class GridSearchAdaptFormer(nn.Module):
    def __init__(self, nb_heads, layer_norm, linear, adapter_activation, reduction_ratios):
        super(GridSearchAdaptFormer, self).__init__()
        self.nb_heads = nb_heads
        if nb_heads == 1:
            self.heads = AdaptFormer(layer_norm, linear, adapter_activation, reduction_ratios[0])
        else:
            self.heads = nn.ModuleList([AdaptFormer(layer_norm, linear, adapter_activation, reduction_ratios[i])
                                         for i in range(nb_heads)])

    def forward(self, x):
        if isinstance(self.heads, AdaptFormer):
            return self.heads(x)
        else:
            bs = x.shape[0] // self.nb_heads
            assert x.shape[0] % self.nb_heads == 0
            x = [self.heads[i](x[i * bs : (i + 1) * bs]) for i in range(self.nb_heads)]
            return torch.cat(x, dim=0)
