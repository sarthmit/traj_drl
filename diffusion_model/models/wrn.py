import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import timeembedding, layers, layerspp
import sys

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, temb_dim=None):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

        self.Dense_0 = nn.Linear(temb_dim, planes)
        self.Dense_0.weight.data = layers.default_init()(self.Dense_0.weight.shape)
        nn.init.zeros_(self.Dense_0.bias)

    def forward(self, x, temb=None):
        out = self.conv1(F.relu(self.bn1(x)))
        if temb is not None:
            temb_bias_wrongshape = self.Dense_0(F.silu(temb))
            temb_bias = temb_bias_wrongshape[:, :, None, None]
            out += temb_bias
        else:
            raise Exception('temb is None')
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, latent_dim, prob_enc=False):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        if k < 1e-8:
            k = 2
            nStages = [16, int(16 * k), int(32 * k), 2]
            latent_dim = 2
        else:
            nStages = [16, int(16 * k), int(32 * k), int(64 * k)]

        self.prob_enc = prob_enc

        self.time_embedding = timeembedding.TimeEmbedding()
        temb_dim = self.time_embedding.temb_dim

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, temb_dim=temb_dim)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, temb_dim=temb_dim)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, temb_dim=temb_dim)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        assert latent_dim == nStages[3]
        if self.prob_enc:
            self.linear_latent = nn.Linear(nStages[3], latent_dim * 2)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, temb_dim=None):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, temb_dim=temb_dim))
            self.in_planes = planes

        return nn.ModuleList(layers)

    def forward(self, x, t=None, use_param_t=False):
        if t is None:
            t = torch.zeros(x.shape[0]).to(x.device)
        temb = self.time_embedding(t, use_param_t=use_param_t)
        out = self.conv1(x)
        for layerlist in [self.layer1, self.layer2, self.layer3]:
            for layer in layerlist:
                out = layer(out, temb=temb)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.shape[-2])
        out = out.view(out.size(0), -1)
        out_resnet = out
        z = out
        if self.prob_enc:
            z = self.linear_latent(z)
        clf_out = self.linear(out)
        return clf_out, out_resnet, z


def build_wideresnet(depth, widen_factor, dropout, num_classes, latent_dim, prob_enc):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return Wide_ResNet(depth=depth,
                       widen_factor=widen_factor,
                       dropout_rate=dropout,
                       num_classes=num_classes,
                       latent_dim=latent_dim,
                       prob_enc=prob_enc)
