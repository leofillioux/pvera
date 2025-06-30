import torch
import numpy as np
import torch.nn as nn
from einops import repeat

class VeRA(nn.Module):
    def __init__(self, linear_layer, in_dim, rank, alpha, q_downsample, v_downsample, q_upsample, v_upsample):
        super(VeRA, self).__init__()
        self.linear_layer = linear_layer
        self.q_downsample = q_downsample
        self.v_downsample = v_downsample
        self.q_upsample = q_upsample
        self.v_upsample = v_upsample
        self.alpha = alpha

        self.adapter_qd = nn.Parameter(torch.ones(rank) * np.random.uniform(1e-5, 1))
        self.adapter_vd = nn.Parameter(torch.ones(rank) * np.random.uniform(1e-5, 1))
        self.adapter_qb = nn.Parameter(torch.zeros(in_dim))
        self.adapter_vb = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        qd = repeat(self.adapter_qd, 'd -> b l d', b=x.shape[0], l=x.shape[1])
        qb = repeat(self.adapter_qb, 'd -> b l d', b=x.shape[0], l=x.shape[1])
        vd = repeat(self.adapter_vd, 'd -> b l d', b=x.shape[0], l=x.shape[1])
        db = repeat(self.adapter_vb, 'd -> b l d', b=x.shape[0], l=x.shape[1])

        x_q = self.alpha * ((x @ self.q_downsample.to(x.device) * qd) @ self.q_upsample.to(x.device) * qb)
        x_v = self.alpha * ((x @ self.v_downsample.to(x.device) * vd) @ self.v_upsample.to(x.device) * db)
        x_lora = torch.cat([x_q, torch.zeros_like(x_v), x_v], dim=-1)
        x = self.linear_layer(x) + x_lora
        return x

class GridSearchVeRA(nn.Module):
    def __init__(self, nb_heads, linear_layer, in_dim, rank, alpha, q_downsample, v_downsample, q_upsample, v_upsample):
        super(GridSearchVeRA, self).__init__()
        self.nb_heads = nb_heads
        if nb_heads == 1:
            self.heads = VeRA(linear_layer, in_dim, rank, alpha, q_downsample[0], v_downsample[0], q_upsample[0], v_upsample[0])
        else:
            self.heads = nn.ModuleList([VeRA(linear_layer, in_dim, rank, alpha,
                                            q_downsample[i], v_downsample[i],
                                            q_upsample[i], v_upsample[i]) for i in range(nb_heads)])

    def forward(self, x):
        if isinstance(self.heads, VeRA):
            x = self.heads(x)
        else:
            bs = x.shape[0] // self.nb_heads
            assert x.shape[0] % self.nb_heads == 0
            x = [self.heads[i](x[i * bs : (i + 1) * bs])
                 for i in range(self.nb_heads)]
            x = torch.concatenate(x, dim=0)
        return x

class PVeRA(nn.Module):
    def __init__(self, linear_layer, in_dim, rank, alpha, q_downsample, v_downsample, q_upsample, v_upsample):
        super(PVeRA, self).__init__()
        self.linear_layer = linear_layer
        self.q_downsample = q_downsample
        self.v_downsample = v_downsample
        self.q_upsample = q_upsample
        self.v_upsample = v_upsample
        self.alpha = alpha

        self.adapter_qd = nn.Parameter(torch.ones(rank*2) * np.random.uniform(1e-5, 1))
        self.adapter_vd = nn.Parameter(torch.ones(rank*2) * np.random.uniform(1e-5, 1))
        self.adapter_qb = nn.Parameter(torch.zeros(in_dim))
        self.adapter_vb = nn.Parameter(torch.zeros(in_dim))

        self.inference_sample = 'mean'

    def reparametrize(self, x, mu, logvar, b, upsample, sample_type='random'):
        sample_type = sample_type if self.training else 'mean'
        kld_loss = 0
        if sample_type == 'random':
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + eps*std
            kld_loss = torch.mean(-0.5 * torch.mean(torch.mean(1 + logvar - mu ** 2 - logvar.exp(), dim=-1), dim=-1), dim=0)
            x = self.alpha * (z @ upsample.to(x.device) * b)
        elif sample_type == 'mean':
            z = mu
            x = self.alpha * (z @ upsample.to(x.device) * b)
        elif sample_type == 'fix':
            std = torch.exp(0.5*logvar)
            z = mu+std
            x = self.alpha * (z @ upsample.to(x.device) * b)
        elif type(sample_type) == int:
            zs = [mu + torch.exp(0.5*logvar)*torch.randn_like(logvar) for _ in range(sample_type)]
            zs = torch.stack(zs)
            x = self.alpha * (zs @ upsample.to(x.device) * b).mean(axis=0)
        return x, kld_loss

    def forward(self, x):
        qd = repeat(self.adapter_qd, 'd -> b l d', b=x.shape[0], l=x.shape[1])
        qb = repeat(self.adapter_qb, 'd -> b l d', b=x.shape[0], l=x.shape[1])
        vd = repeat(self.adapter_vd, 'd -> b l d', b=x.shape[0], l=x.shape[1])
        vb = repeat(self.adapter_vb, 'd -> b l d', b=x.shape[0], l=x.shape[1])

        # reparametrize q
        mu_q, logvar_q = (x @ self.q_downsample.to(x.device) * qd).chunk(2, dim=-1)
        x_q, kld_q = self.reparametrize(x, mu_q, logvar_q, qb, self.q_upsample)

        # reparametrize v
        mu_v, logvar_v = (x @ self.v_downsample.to(x.device) * vd).chunk(2, dim=-1)
        x_v, kld_v = self.reparametrize(x, mu_v, logvar_v, vb, self.v_upsample)

        x_lora = torch.cat([x_q, torch.zeros_like(x_v), x_v], dim=-1)
        x = self.linear_layer(x) + x_lora
        self.kld = (kld_q+kld_v)/2
        return x

class GridSearchPVeRA(nn.Module):
    def __init__(self, nb_heads, linear_layer, in_dim, rank, alpha, q_downsample, v_downsample, q_upsample, v_upsample):
        super(GridSearchPVeRA, self).__init__()
        self.nb_heads = nb_heads
        if nb_heads == 1:
            self.heads = PVeRA(linear_layer, in_dim, rank, alpha, q_downsample[0], v_downsample[0], q_upsample[0], v_upsample[0])
        else:
            self.heads = nn.ModuleList([PVeRA(linear_layer, in_dim, rank, alpha, q_downsample[i], v_downsample[i], q_upsample[i], v_upsample[i])
                                        for i in range(nb_heads)])

    def forward(self, x):
        if isinstance(self.heads, PVeRA):
            x = self.heads(x)
        else:
            bs = x.shape[0] // self.nb_heads
            assert x.shape[0] % self.nb_heads == 0
            x = [self.heads[i](x[i * bs : (i + 1) * bs])
                 for i in range(self.nb_heads)]
            x = torch.concatenate(x, dim=0)
        return x
