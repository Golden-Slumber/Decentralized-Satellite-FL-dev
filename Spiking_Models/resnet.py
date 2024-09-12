import torch
import torch.nn as nn
from Spiking_Models.layer import *


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def warpBN(channel, bn_type, nb_steps):
    if bn_type == 'tdbn':
        return tdLayer(nn.BatchNorm2d(channel), nb_steps)
    elif bn_type == 'bntt':
        return TemporalBN(channel, nb_steps, step_wise=True)
    elif bn_type == '':
        return TemporalBN(channel, nb_steps, step_wise=False)
    elif bn_type == 'idnt':
        return nn.Identity()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_type='', expand=1, **kwargs_spikes):
        super(BasicBlock, self).__init__()
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']
        self.expand = expand
        self.conv1 = tdLayer(nn.Conv2d(in_planes, planes * expand, kernel_size=3, stride=stride, padding=1, bias=False),
                             self.nb_steps)
        self.bn1 = warpBN(planes * expand, bn_type, self.nb_steps)
        self.spike1 = LIFLayer(**kwargs_spikes)
        self.conv2 = tdLayer(nn.Conv2d(planes, planes * expand, kernel_size=3, stride=1, padding=1, bias=False),
                             self.nb_steps)
        self.bn2 = warpBN(planes * expand, bn_type, self.nb_steps)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(tdLayer(
                nn.Conv2d(in_planes, planes * self.expansion * expand, kernel_size=1, stride=stride, bias=False),
                self.nb_steps),
                warpBN(self.expansion * planes * expand, bn_type, self.nb_steps))
        self.spike2 = LIFLayer(**kwargs_spikes)

    def forward(self, x):
        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.spike2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_type='', **kwargs_spikes):
        super(Bottleneck, self).__init__()
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']
        self.conv1 = tdLayer(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False), self.nb_steps)
        self.bn1 = warpBN(planes, bn_type, self.nb_steps)
        self.spike1 = LIFLayer(**kwargs_spikes)
        self.conv2 = tdLayer(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                             self.nb_steps)
        self.bn2 = warpBN(planes, bn_type, self.nb_steps)
        self.spike2 = LIFLayer(**kwargs_spikes)
        self.conv3 = tdLayer(nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False), self.nb_steps)
        self.bn3 = warpBN(self.expansion * planes, bn_type, self.nb_steps)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                tdLayer(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                        self.nb_steps),
                warpBN(self.expansion * planes, bn_type, self.nb_steps)
            )
        self.spike3 = LIFLayer(**kwargs_spikes)

    def forward(self, x):
        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.spike2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.spike3(out)
        return out