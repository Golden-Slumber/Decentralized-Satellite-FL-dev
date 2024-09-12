import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from Spiking_Models.activation import NoisySpike


class LIFNeuron(nn.Module):
    def __init__(self, spike_fn, decay=None, threshold=None, vreset=None):
        super(LIFNeuron, self).__init__()
        self.decay = copy.deepcopy(decay)
        self.threshold = copy.deepcopy(threshold)
        self.vreset = copy.deepcopy(vreset)
        self.spike_fn = copy.deepcopy(spike_fn)

    def _reset_parameters(self):
        if self.threshold is None:
            self.threshold = 0.5
        if self.decay is None:
            self.decay = nn.Parameter(torch.Tensor([0.9]))

    def forward(self, vmem, psp):
        vmem = torch.sigmoid(self.decay) * vmem + psp
        if isinstance(self.spike_fn, NoisySpike):
            self.spike_fn.reset_mask()
        spike = self.spike_fn(vmem - self.threshold)
        if self.vreset is None:
            vmem -= self.threshold * spike
        else:
            vmem = vmem * (1 - spike) + self.vreset * spike
        return vmem, spike
