import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        out = self.relu(out)
        return out


class Network(nn.Module):
    def __init__(self, block=BasicBlock):
        super(Network, self).__init__()
        self.inplanes = 64
        self.num_layers = 24

        self.inc = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self._make(block, self.inplanes)

        self.outc = nn.Conv2d(self.inplanes, 3, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Fixup init (only scale)
                nn.init.normal_(m.weight, mean=0, std=np.sqrt(2 / (m.weight.shape[0] * np.prod(m.weight.shape[2:]))) * self.num_layers ** (-0.5))
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make(self, block, planes):
        for i in range(self.num_layers):
            meta = block(self.inplanes, planes)
            setattr(self, 'layer{}'.format(i), meta)

    def forward(self, x, pretrain=False):
        x = self.inc(x)
        identity = x

        for i in range(self.num_layers):
            x = getattr(self, 'layer{}'.format(i))(x)

        x = x + identity
        x = self.outc(x)

        return x
        