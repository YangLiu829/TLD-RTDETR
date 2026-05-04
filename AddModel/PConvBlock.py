import torch
import torch.nn as nn
from collections import OrderedDict
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, k, s, act='relu'):
        super().__init__()
        padding = k // 2
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def get_activation(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'silu':
        return nn.SiLU()
    else:
        return nn.Identity()
class Partial_conv3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return self.conv(x)


# ====== BasicBlock ======
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)

        short = x if self.shortcut else self.short(x)

        out = out + short
        out = self.act(out)

        return out


# ====== PConvBlock ======
class PConvBlock(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)

        # 替换 branch2b
        self.branch2b = nn.Sequential(
            Partial_conv3(dim=ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )



