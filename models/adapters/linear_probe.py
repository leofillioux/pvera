import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x, activation=True):
        x = self.linear(x)
        if activation:
            x = self.activation(x)
        return x

class GridSearchLinearProbe(nn.Module):
    def __init__(self, nb_heads, input_dim, output_dim):
        super(GridSearchLinearProbe, self).__init__()
        self.nb_heads = nb_heads
        if nb_heads == 1:
            self.heads = LinearProbe(input_dim, output_dim)
        else:
            self.heads = nn.ModuleList([LinearProbe(input_dim, output_dim) for _ in range(nb_heads)])
    
    def forward(self, x, activation=True):
        if isinstance(self.heads, LinearProbe):
            x = self.heads(x, activation)
        else:
            bs = x.shape[0] // self.nb_heads
            assert x.shape[0] % self.nb_heads == 0
            x = [self.heads[i](x[i * bs : (i + 1) * bs], activation)
                 for i in range(self.nb_heads)]
        return x