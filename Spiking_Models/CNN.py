import torch
import torch.nn as nn
import pickle
from Spiking_Models.layer import *
from torchvision.models.resnet import BasicBlock

def warpBN(channel, bn_type, nb_steps):
    if bn_type == 'tdbn':
        return tdLayer(nn.BatchNorm2d(channel), nb_steps)
    elif bn_type == 'bntt':
        return TemporalBN(channel, nb_steps, step_wise=True)
    elif bn_type == '':
        return TemporalBN(channel, nb_steps, step_wise=False)
    elif bn_type == 'idnt':
        return nn.Identity()


class SpikingCNN(nn.Module):
    def __init__(self, num_classes=10, bn_type='', **kwargs_spikes):
        super(SpikingCNN, self).__init__()
        self.bn_type = bn_type
        self.kwargs_spikes = kwargs_spikes
        self.nb_steps = kwargs_spikes['nb_steps']

        self.conv1 = nn.Sequential(
            tdLayer(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, stride=1, bias=False),
                    nb_steps=self.nb_steps),
            warpBN(4, bn_type, self.nb_steps),
            LIFLayer(**kwargs_spikes)
        )
        self.maxpool1 = tdLayer(nn.MaxPool2d(kernel_size=2), nb_steps=self.nb_steps)

        self.conv2 = nn.Sequential(
            tdLayer(nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1, bias=False),
                    nb_steps=self.nb_steps),
            warpBN(8, bn_type, self.nb_steps),
            LIFLayer(**kwargs_spikes)
        )
        self.maxpool2 = tdLayer(nn.MaxPool2d(kernel_size=2), nb_steps=self.nb_steps)

        self.fc = tdLayer(nn.Linear(in_features=8 * 16 * 16, out_features=32), nb_steps=self.nb_steps)
        self.classifier = nn.Sequential(
            tdLayer(nn.Linear(in_features=32, out_features=num_classes), nb_steps=self.nb_steps),
            ReadOut()
        )

    def forward(self, x):
        out, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape))
        out = self.conv1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out