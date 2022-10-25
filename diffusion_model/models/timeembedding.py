import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers, layerspp

logger = logging.getLogger(__name__)
default_initializer = layers.default_init


class TimeEmbedding(nn.Module):
    def __init__(self):
        super(TimeEmbedding, self).__init__()
        self.param_t = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.sigma_min = 0.01
        self.sigma_max = 50.0
        nf = 16
        self.nf = nf
        self.fourier_scale = 16
        embed_dim = 2 * nf
        self.temb_dim = nf * 4

        modules = []
        modules.append(layerspp.GaussianFourierProjection(embedding_size=self.nf, scale=self.fourier_scale))
        modules.append(nn.Linear(embed_dim, nf * 4))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(nf * 4, self.temb_dim))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        self.all_modules = nn.ModuleList(modules)

    def forward(self, t, use_param_t=False):
        if use_param_t:
            t = t * 0. + torch.sigmoid(self.param_t)
        time_cond = self.sigma_min * (self.sigma_max / self.sigma_min) ** t

        modules = self.all_modules
        m_idx = 0

        used_sigmas = time_cond
        temb = modules[m_idx](torch.log(used_sigmas))
        m_idx += 1
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](F.silu(temb))
        m_idx += 1

        return temb
