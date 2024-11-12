import copy
import sys
import numpy
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from utils import Dirichlet_non_iid_distribution
import pickle
from Spiking_Models.activation import NoisySpike, InvSigmoid, InvRectangle
from Spiking_Models.neuron import LIFNeuron
from Spiking_Models.resnet import SpikingBasicBlock, SmallResNet, ArtificialSmallResnet
from constants import *


def wrap_decay(decay):
    import math
    return torch.tensor(math.log(decay / (1 - decay)))


def split_params(model, paras=([], [], [])):
    for n, module in model._modules.items():
        if isinstance(module, LIFNeuron) and hasattr(module, "thresh"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'bathnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd):
            paras[1].append(module.weight)
            if module.bias is not None:
                paras[2].append(module.bias)
        elif len(list(module.children())) > 0:
            paras = split_params(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras


def average_weights(weight_list, aggregation_weights=None):
    w_avg = deepcopy(weight_list[0])
    for key in w_avg.keys():
        if aggregation_weights is None:
            for i in range(1, len(weight_list)):
                w_avg[key] += deepcopy(weight_list[i][key])
            w_avg[key] = torch.div(w_avg[key], len(weight_list))
        else:
            w_avg[key] = w_avg[key] * aggregation_weights[0]
            for i in range(1, len(weight_list)):
                w_avg[key] += deepcopy(weight_list[i][key]) * aggregation_weights[i]
    return w_avg


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone().detach(), label.clone().detach()


class ConstellationLearning(object):
    def __init__(self, num_planes, num_sat_by_planes, train_dataset, test_dataset, args):
        print('constellation initialization...')
        # satellite constellation configuration
        self.num_planes = num_planes
        self.num_sat_by_planes = num_sat_by_planes

        # learning parameters
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.global_weight = None
        self.local_weights = None
        self.intra_plane_weights = None
        self.local_dataset_indices = [[[] for j in range(self.num_sat_by_planes[i])] for i in range(self.num_planes)]
        self.dtype = torch.float
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = None
        self.optimizer = None
        self.width_optim = None

        # aggregation parameters
        self.aggregation_matrix = None
        self.aggregation_scheme = ALLREDUCE
        self.received_relay_weights = None
        self.transmitted_relay_weights = None
        self.received_relay_counts = None
        self.transmitted_relay_counts = None

        # performance metrics
        self.convergence_error = numpy.zeros(self.args.num_epoch)
        self.consensus_error = numpy.zeros(self.args.num_epoch)
        self.test_accuracy = numpy.zeros(self.args.num_epoch)

    def dataset_partition(self):
        print('dataset partition...')
        # non-IID inter-plane partition
        # targets = []
        # for i in range(len(self.train_dataset)):
        #     _, target = self.train_dataset[i]
        #     targets.append(target.item())
        # indices_per_plane = Dirichlet_non_iid_distribution(targets, self.args.plane_alpha, self.num_planes,
        #                                                    n_auxi_devices=10, seed=0)
        # for i in range(self.num_planes):
        #     num_samples_per_satellite = int(len(indices_per_plane[i]) / self.num_sat_by_planes[i])
        #     plane_indices = deepcopy(indices_per_plane[i])
        #     for j in range(self.num_sat_by_planes[i]):
        #         self.local_dataset_indices[i][j] = numpy.random.choice(plane_indices, num_samples_per_satellite,
        #                                                                replace=False)
        #         plane_indices = list(set(plane_indices) - set(self.local_dataset_indices[i][j]))

        # IID inter-plane partition
        all_indices = [i for i in range(len(self.train_dataset))]
        num_samples_per_plane = int(len(all_indices) / self.num_planes)
        for i in range(self.num_planes):
            plane_indices = numpy.random.choice(all_indices, num_samples_per_plane, replace=False)
            all_indices = list(set(all_indices) - set(plane_indices))

            num_samples_per_satellite = int(len(plane_indices) / self.num_sat_by_planes[i])
            for j in range(self.num_sat_by_planes[i]):
                self.local_dataset_indices[i][j] = numpy.random.choice(plane_indices, num_samples_per_satellite,
                                                                       replace=False)
                plane_indices = list(set(plane_indices) - set(self.local_dataset_indices[i][j]))

        # for i in range(self.num_planes):
        #     for j in range(self.num_sat_by_planes[i]):
        #         print('{}, {}'.format(i, j))
        #         print(self.local_dataset_indices[i][j])

    def spike_learning_initialization(self):
        print('spike learning initialization...')
        decay = nn.Parameter(wrap_decay(self.args.decay))
        thresh = self.args.thresh
        alpha = 1 / self.args.alpha
        if self.args.act == 'mns_rec':
            inv_sg = InvRectangle(alpha=alpha, learnable=self.args.train_width, granularity=self.args.granularity)
        elif self.args.act == 'mns_sig':
            inv_sg = InvSigmoid(alpha=alpha, learnable=self.args.train_width)
        kwargs_spikes = {'nb_steps': self.args.T, 'vreset': 0, 'threshold': thresh,
                         'spike_fn': NoisySpike(p=self.args.p, inv_sg=inv_sg, spike=True), 'decay': decay}

        self.model = SmallResNet(SpikingBasicBlock, [1, 2, 2, 2], num_classes=10, bn_type=self.args.bn_type,
                                 **kwargs_spikes).to(self.device, self.dtype)
        self.model(self.train_dataset[0][0].unsqueeze(0).to(self.device, self.dtype))
        self.global_weight = self.model.state_dict()
        self.local_weights = [[deepcopy(self.global_weight) for j in range(self.num_sat_by_planes[i])] for i in
                              range(self.num_planes)]
        self.intra_plane_weights = [deepcopy(self.global_weight) for i in range(self.num_planes)]

        params = split_params(self.model)
        spiking_params = [{'params': params[0], 'weight_decay': 0}]
        params = [{'params': params[1], 'weight_decay': self.args.wd}, {'params': params[2], 'weight_decay': 0}]
        self.optimizer = optim.SGD(params, lr=self.args.lr, momentum=0.9)
        self.width_optim = optim.Adam(spiking_params, lr=self.args.width_lr)

    def inter_plane_aggregation_configuration(self, aggregation_matrix, aggregation_scheme):
        print('inter-plane aggregation configuration...')
        self.aggregation_matrix = aggregation_matrix
        self.aggregation_scheme = aggregation_scheme

        if aggregation_scheme == RELAYSUM:
            self.received_relay_weights = [[] for i in range(self.num_planes)]
            self.transmitted_relay_weights = [[] for i in range(self.num_planes)]
            self.received_relay_counts = [[] for i in range(self.num_planes)]
            self.transmitted_relay_counts = [[] for i in range(self.num_planes)]
            tmp_model = deepcopy(self.global_weight)
            zero_model = deepcopy(self.global_weight)
            for key in tmp_model.keys():
                zero_model[key] -= tmp_model[key]
            for plane_idx in range(self.num_planes):
                for neighbor_idx in range(self.num_planes):
                    self.received_relay_weights[plane_idx].append(deepcopy(zero_model))
                    self.transmitted_relay_weights[plane_idx].append(deepcopy(zero_model))
                    self.received_relay_counts[plane_idx].append(0)
                    self.transmitted_relay_counts[plane_idx].append(0)

    def training(self, epoch, plane_iter, plane_idx, sat_idx):
        self.model.load_state_dict(self.local_weights[plane_idx][sat_idx])
        self.model.train()

        lr = self.args.lr * (1 + math.cos(math.pi * epoch / self.args.num_epoch)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        data_loader = DataLoader(DatasetSplit(self.train_dataset, self.local_dataset_indices[plane_idx][sat_idx]),
                                 batch_size=self.args.train_batch_size, shuffle=True,
                                 pin_memory=True, num_workers=self.args.num_workers)
        for local_iter in range(self.args.local_iters):
            loss_tot = []
            predict_tot = []
            label_tot = []

            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1)

                self.optimizer.zero_grad()
                self.width_optim.zero_grad()

                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.width_optim.step()

                predict = torch.argmax(output, dim=1)
                loss_tot.append(loss.detach().cpu())
                predict_tot.append(predict)
                label_tot.append(target)
            predict_tot = torch.cat(predict_tot)
            label_tot = torch.cat(label_tot)
            local_train_acc = torch.mean((predict_tot == label_tot).float())
            local_train_loss = torch.tensor(loss_tot).sum() / len(label_tot)
            print(
                '\t Plane {} Sat {}, Epoch [{}/{}], Plane Iter [{}/{}], Local Iter [{}/{}] Local Loss: {:.5f}, Local Acc: {:.5f}'.format(
                    plane_idx, sat_idx, epoch + 1,
                    self.args.num_epoch, plane_iter + 1,
                    self.args.intra_plane_iters, local_iter + 1,
                    self.args.local_iters,
                    local_train_loss,
                    local_train_acc))
        self.local_weights[plane_idx][sat_idx] = deepcopy(self.model.state_dict())

    def inference(self, data_loader):
        self.model.load_state_dict(self.global_weight)
        self.model.eval()

        with torch.no_grad():
            predict_tot = []
            label_tot = []
            loss_tot = []

            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1)
                output = self.model(data)

                loss = self.criterion(output, target)
                predict = torch.argmax(output, dim=1)
                predict_tot.append(predict)
                loss_tot.append(loss)
                label_tot.append(target)

            label_tot = torch.cat(label_tot)
            test_loss = torch.tensor(loss_tot).sum() / len(label_tot)
            predict_tot = torch.cat(predict_tot)
            test_acc = torch.mean((predict_tot == label_tot).float())

            return test_loss, test_acc

    def constellation_learning(self, epoch):
        for plane_iter in range(self.args.intra_plane_iters):
            # local training
            for plane_idx in range(self.num_planes):
                for sat_idx in range(self.num_sat_by_planes[plane_idx]):
                    self.training(epoch, plane_iter, plane_idx, sat_idx)
            # intra-plane aggregation
            for plane_idx in range(self.num_planes):
                self.intra_plane_weights[plane_idx] = average_weights(self.local_weights[plane_idx])
                for sat_idx in range(self.num_sat_by_planes[plane_idx]):
                    self.local_weights[plane_idx][sat_idx] = deepcopy(self.intra_plane_weights[plane_idx])

        # inter-plane aggregation
        new_intra_plane_weights = []
        if self.aggregation_scheme == GOSSIP:
            aggregation_matrix = numpy.zeros((self.num_planes, self.num_planes))
            for p in range(self.num_planes):
                if p == 0:
                    aggregation_matrix[p] = 1 / 2
                    aggregation_matrix[p + 1] = 1 / 2
                elif p == self.num_planes - 1:
                    aggregation_matrix[p] = 1 / 2
                    aggregation_matrix[p - 1] = 1 / 2
                else:
                    aggregation_matrix[p] = 1 / 3
                    aggregation_matrix[p - 1] = 1 / 3
                    aggregation_matrix[p + 1] = 1 / 3

            for plane_idx in range(self.num_planes):
                new_intra_plane_weights.append(
                    average_weights(self.intra_plane_weights, list(aggregation_matrix[plane_idx])))

        elif self.aggregation_scheme == RELAYSUM:
            for plane_idx in range(self.num_planes):
                # find neighbors to relay messages stored
                for neighbor_idx in range(self.num_planes):
                    if self.aggregation_matrix[plane_idx, neighbor_idx] == 1 and plane_idx != neighbor_idx:
                        new_relay_weights = deepcopy(self.intra_plane_weights[plane_idx])
                        new_relay_count = 1
                        # calculate the messages that will be relayed to this neighbor
                        for inner_idx in range(self.num_planes):
                            if self.aggregation_matrix[
                                inner_idx, plane_idx] == 1 and inner_idx != neighbor_idx and inner_idx != plane_idx:
                                for key in new_relay_weights.keys():
                                    new_relay_weights[key] += self.received_relay_weights[plane_idx][inner_idx][key]
                                new_relay_count += self.received_relay_counts[plane_idx][inner_idx]
                        self.transmitted_relay_weights[plane_idx][neighbor_idx] = deepcopy(new_relay_weights)
                        self.transmitted_relay_counts[plane_idx][neighbor_idx] = new_relay_count
            # replay messages and inter-plane model aggregation
            for plane_idx in range(self.num_planes):
                plane_weights = deepcopy(self.intra_plane_weights[plane_idx])
                plane_count = 1
                for neighbor_idx in range(self.num_planes):
                    if self.aggregation_matrix[neighbor_idx, plane_idx] == 1 and plane_idx != neighbor_idx:
                        self.received_relay_weights[plane_idx][neighbor_idx] = deepcopy(
                            self.transmitted_relay_weights[neighbor_idx][plane_idx])
                        self.received_relay_counts[plane_idx][neighbor_idx] = \
                            self.transmitted_relay_counts[neighbor_idx][plane_idx]
                        plane_count += self.received_relay_counts[plane_idx][neighbor_idx]
                        for key in plane_weights.keys():
                            plane_weights[key] += self.received_relay_weights[plane_idx][neighbor_idx][key]
                for key in plane_weights.keys():
                    plane_weights[key] /= plane_count
                new_intra_plane_weights.append(plane_weights)
        elif self.aggregation_scheme == ALLREDUCE:
            print('all reduce')
            all_reduced_weights = average_weights(self.intra_plane_weights)
            for plane_idx in range(self.num_planes):
                new_intra_plane_weights.append(deepcopy(all_reduced_weights))

        self.global_weight = average_weights(new_intra_plane_weights)
        for plane_idx in range(self.num_planes):
            for sat_idx in range(self.num_sat_by_planes[plane_idx]):
                self.local_weights[plane_idx][sat_idx] = deepcopy(new_intra_plane_weights[plane_idx])

        # performance metric recording
        training_loader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                     pin_memory=True, num_workers=self.args.num_workers)
        loss, _ = self.inference(training_loader)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.test_batch_size, shuffle=True,
                                 pin_memory=True, num_workers=self.args.num_workers)
        _, acc = self.inference(test_loader)

        self.convergence_error[epoch] = loss
        self.test_accuracy[epoch] = acc
        print('global round : {} \t Loss: {:.6f} \t Accuracy: {:.3f}%'.format(epoch, loss, 100. * acc))


if __name__ == '__main__':
    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    print(train_set[0][0].shape)
    print(train_set[0][0].unsqueeze(0).shape)

    targets = []
    for i in range(len(train_set)):
        _, target = train_set[i]
        targets.append(target.item())
    indices_per_plane = Dirichlet_non_iid_distribution(targets, 0.2, 9, n_auxi_devices=10, seed=0)
    print(type(indices_per_plane))
    print(indices_per_plane)
