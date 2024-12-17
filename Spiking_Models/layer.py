import torch
import torch.nn as nn
from Spiking_Models.neuron import LIFNeuron


class LIFLayer(nn.Module):
    def __init__(self, neuron=LIFNeuron, nb_steps=0, **neuron_args):
        super(LIFLayer, self).__init__()
        assert nb_steps > 0
        self.neuron = neuron(**neuron_args)
        self.nb_steps = nb_steps

        self.avg_spike_rate = None

    def create_mask(self, x: torch.Tensor, p: float):
        return torch.bernoulli(torch.ones_like(x) * (1 - p))

    def forward(self, x):
        vmem = 0
        spikes = []
        for step in range(self.nb_steps):
            current = x[step]
            vmem, spike = self.neuron(vmem, current)
            spikes.append(spike * self.neuron.threshold)
        # tmp_spikes = torch.stack(spikes).view(3, -1)
        # print(tmp_spikes.shape)
        # total_spikes = 0
        # for i in range(3):
        #     for j in range(tmp_spikes.shape[1]):
        #         if tmp_spikes[i, j] == 1:
        #             total_spikes += 1
        # print(total_spikes)
        # print(tmp_spikes)
        if not self.training:
            self.record_spikes(torch.stack(spikes).view(self.nb_steps, x.shape[1], -1))
        return torch.stack(spikes)

    def record_spikes(self, spikes):
        num_spikes = 0
        for idx in range(spikes.shape[1]):
            num_spikes += torch.sum(spikes[:, idx, :])
        self.avg_spike_rate = num_spikes / spikes.shape[1] / spikes.shape[2]

class tdLayer(nn.Module):
    def __init__(self, layer, nb_steps):
        super(tdLayer, self).__init__()
        self.nb_steps = nb_steps
        self.layer = layer

    def forward(self, x):
        x = x.contiguous()
        x = self.layer(x.view(-1, *x.shape[2:]))
        return x.view(self.nb_steps, -1, *x.shape[1:])


class tbBatchNorm(nn.Module):
    def __init__(self, bn, alpha=1, Vth=0.5):
        super(tbBatchNorm, self).__init__()
        self.bn = bn
        self.alpha = alpha
        self.Vth = Vth

    def forward(self, x):
        exponential_average_factor = 0.0

        if self.training and self.bn.track_running_stats:
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked += 1
                if self.bn.momentum is None:
                    exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
                else:
                    exponential_average_factor = self.bn.momentum

        if self.training:
            mean = x.mean([0, 2, 3, 4], keepdim=True)
            var = x.var([0, 2, 3, 4], keepdim=-True, unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.bn.running_mean = exponential_average_factor * mean[0, :, 0, 0, 0] + (
                        1 - exponential_average_factor) * self.bn.runi
                self.bn.running_var = exponential_average_factor * var[0, :, 0, 0, 0] * n / (n - 1) + (
                        1 - exponential_average_factor) * self.bn.running_var
        else:
            mean = self.bn.running_mean[None, :, None, None, None]
            var = self.bn.running_var[None, :, None, None, None]

        x = self.alpha * self.Vth * (x - mean) / (torch.sqrt(var) + self.bn.eps)

        if self.bn.affine:
            x = x * self.bn.weight[None, :, None, None, None] + self.bn.bias[None, :, None, None, None]

        return x


class TemporalBN(nn.Module):
    def __init__(self, in_channels, nb_steps, step_wise=False):
        super(TemporalBN, self).__init__()
        self.nb_steps = nb_steps
        self.step_wise = step_wise
        if not step_wise:
            self.bns = nn.BatchNorm2d(in_channels)
        else:
            self.bns = nn.ModuleList([nn.BatchNorm2d(in_channels) for t in range(self.nb_steps)])

    def forward(self, x):
        out = []
        for step in range(self.nb_steps):
            if self.step_wise:
                out.append(self.bns[step](x[step]))
            else:
                out.append(self.bns(x[step]))
        out = torch.stack(out)
        return out


class ReadOut(nn.Module):
    def __init__(self, mode='psp_avg', neuron=None, neuron_args=None):
        super(ReadOut, self).__init__()
        self.mode = mode
        if 'vmem' in mode:
            self.neuron = neuron(**neuron_args)

    def forward(self, x):
        trace = x
        if 'max' in self.mode:
            out, _ = torch.max(trace, dim=0)
        elif 'avg' in self.mode:
            out = torch.mean(trace, dim=0)
        return out
