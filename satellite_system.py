import os
import random
import sys
import numpy
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
from learning_task import EuroSatTask
from constants import *

home_dir = './'
sys.path.append(home_dir)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def model_aggregation(model_list, average_weights):
    aggregated_model = deepcopy(model_list[0])
    for key in aggregated_model.keys():
        aggregated_model[key] = aggregated_model[key] * average_weights[0]
        for i in range(1, len(model_list)):
            aggregated_model[key] += deepcopy(model_list[i][key]) * average_weights[i]
    return aggregated_model


class Satellite(object):
    def __init__(self, satellite_idx, plane_idx, learning_task):
        self.satellite_idx = satellite_idx
        self.plane_idx = plane_idx
        self.learning_task = deepcopy(learning_task)


class Plane(object):
    def __init__(self, plane_idx, num_satellites, plane_dataset, datasize_by_satellite, init_model, args):
        self.plane_idx = plane_idx
        self.num_satellites = num_satellites
        self.plane_dataset = deepcopy(plane_dataset)
        self.datasize_by_satellite = deepcopy(datasize_by_satellite)
        self.sum_weight_arr = deepcopy(datasize_by_satellite) / numpy.sum(datasize_by_satellite)
        self.intra_plane_model = deepcopy(init_model)
        self.args = args

        self.satellite_list = []
        total_sample_idxs = [i for i in range(len(plane_dataset))]
        for satellite_idx in range(num_satellites):
            local_sample_idx = set(
                numpy.random.choice(total_sample_idxs, int(datasize_by_satellite[satellite_idx]), replace=False))
            total_sample_idxs = list(set(total_sample_idxs) - local_sample_idx)

            new_learning_task = EuroSatTask(args, DatasetSplit(self.plane_dataset, local_sample_idx),
                                            deepcopy(init_model))
            new_satellite = Satellite(satellite_idx, self.plane_idx, new_learning_task)
            self.satellite_list.append(new_satellite)

        # relay model messages for RelaySum aggregation
        self.received_relay_models = []
        self.received_relay_counts = []
        self.transmitted_relay_models = []
        self.transmitted_relay_counts = []

    def relay_sum_initialization(self, num_planes):
        tmp_model = deepcopy(self.intra_plane_model)
        zero_model = deepcopy(self.intra_plane_model)
        for key in tmp_model.keys():
            zero_model[key] -= tmp_model[key]
        for p in range(num_planes):
            self.received_relay_models.append(zero_model)
            self.received_relay_counts.append(0)

    def get_satellites(self):
        return self.satellite_list

    def get_intra_plane_model(self):
        return self.intra_plane_model

    def get_received_message(self, idx):
        return self.received_relay_counts[idx], self.received_relay_models[idx]

    def set_received_message(self, idx, new_relay_count, new_relay_model):
        self.received_relay_models[idx] = deepcopy(new_relay_model)
        self.received_relay_counts[idx] = new_relay_count

    def get_transmitted_message(self, idx):
        return self.transmitted_relay_counts[idx], self.transmitted_relay_models[idx]

    def set_transmitted_message(self, idx, new_relay_count, new_relay_model):
        self.transmitted_relay_models[idx] = deepcopy(new_relay_model)
        self.transmitted_relay_counts[idx] = new_relay_count

    def set_intra_plane_model(self, update_model):
        self.intra_plane_model = deepcopy(update_model)
        for satellite in self.satellite_list:
            satellite.learning_task.model_update(update_model)

    def intra_plane_training(self):
        for satellite in self.satellite_list:
            satellite.learning_task.local_training()

    def intra_plane_model_aggregation(self):
        local_models = []
        for satellite in self.satellite_list:
            local_models.append(satellite.learning_task.get_model())
        intra_plane_model = deepcopy(local_models[0])
        for key in intra_plane_model.keys():
            intra_plane_model[key] = intra_plane_model[key] * self.sum_weight_arr[0]
            for i in range(1, self.num_satellites):
                intra_plane_model[key] += deepcopy(local_models[i][key]) * self.sum_weight_arr[i]
        self.intra_plane_model = deepcopy(intra_plane_model)
        return intra_plane_model


class Constellation(object):
    def __init__(self, num_planes, satellites_by_plane, constellation_dataset, test_dataset, datasize_by_plane,
                 init_model, args):
        self.num_planes = num_planes
        self.satellites_by_plane = deepcopy(satellites_by_plane)
        self.constellation_dataset = deepcopy(constellation_dataset)
        self.test_dataset = test_dataset
        self.datasize_by_plane = deepcopy(datasize_by_plane)
        self.sum_weight_arr = numpy.array(list(map(sum, datasize_by_plane))) / sum(map(sum, datasize_by_plane))
        self.args = args
        self.connectivity_matrix = None
        self.global_model = deepcopy(init_model)

        # performance metrics
        self.convergence_error = numpy.zeros(self.args.iterations)
        self.consensus_error = numpy.zeros(self.args.iterations)
        self.test_accuracy = numpy.zeros(self.args.iterations)
        self.test_task = EuroSatTask(args, test_dataset, deepcopy(init_model))

        self.plane_list = []
        total_sample_idxs = [i for i in range(len(constellation_dataset))]
        for plane_idx in range(num_planes):
            plane_sample_idx = set(
                numpy.random.choice(total_sample_idxs, int(sum(datasize_by_plane[plane_idx])), replace=False))
            total_sample_idxs = list(set(total_sample_idxs) - plane_sample_idx)

            new_plane = Plane(plane_idx, self.satellites_by_plane[plane_idx],
                              DatasetSplit(self.constellation_dataset, plane_sample_idx), datasize_by_plane[plane_idx],
                              deepcopy(init_model), args)
            self.plane_list.append(new_plane)

    def set_connectivity_matrix(self, connectivity_matrix):
        self.connectivity_matrix = connectivity_matrix

    def constellation_training(self, aggregation_scheme):
        intra_plane_model_list = []
        for plane_iter in range(self.args.intra_plane_iters):
            for plane in self.plane_list:
                plane.intra_plane_training()
                plane.intra_plane_model_aggregation()
        for plane in self.plane_list:
            intra_plane_model_list.append(deepcopy(plane.get_intra_plane_model()))

        self.global_model = deepcopy(intra_plane_model_list[0])
        for key in self.global_model.keys():
            self.global_model[key] /= self.num_planes
            for i in range(1, self.num_planes):
                self.global_model[key] += intra_plane_model_list[i] / self.num_planes

        if aggregation_scheme == GOSSIP:
            # generate mixing matrix
            average_matrix = numpy.ones((self.num_planes, self.num_planes))
            for p in range(self.num_planes):
                num_neighbors = numpy.sum(self.connectivity_matrix[p])
                average_matrix[p] = self.connectivity_matrix[p] / num_neighbors

            for p in range(self.num_planes):
                plane_model = model_aggregation(intra_plane_model_list, list(average_matrix[p]))
                self.plane_list[p].set_intra_plane_model(plane_model)
        elif aggregation_scheme == RELAYSUM:
            for p in range(self.num_planes):
                # find neighbors to relay messages in the previous round
                for neighbor in range(self.num_planes):
                    if self.connectivity_matrix[p, neighbor] == 1:
                        new_relay_model = intra_plane_model_list[p]
                        new_relay_count = 1
                        # calculate the messages that will be relayed to neighbor
                        for inner_p in range(self.num_planes):
                            if self.connectivity_matrix[inner_p, p] == 1 and inner_p != neighbor:
                                old_relay_count, old_relay_model = self.plane_list[p].get_received_message(inner_p)
                                for key in new_relay_model.keys():
                                    new_relay_model[key] += old_relay_model[key]
                                new_relay_count += old_relay_count
                        self.plane_list[p].set_transmitted_message(neighbor, new_relay_count, new_relay_model)
            # relay messages & inter-plane model aggregation
            for p in range(self.num_planes):
                plane_model = deepcopy(intra_plane_model_list[p])
                plane_count = 1
                for neighbor in range(self.num_planes):
                    if self.connectivity_matrix[neighbor, p] == 1:
                        relay_count, relay_model = self.plane_list[neighbor].get_transmitted_message(p)
                        self.plane_list[p].set_received_message(neighbor, relay_count, relay_model)
                        plane_count += relay_count
                        for key in plane_model.keys():
                            plane_model[key] += relay_model[key]
                for key in plane_model.keys():
                    plane_model[key] /= plane_count
                self.plane_list[p].set_intra_plane_model(plane_model)

    def save_metric(self, t):
        self.test_task.model_update(self.global_model)
        acc, loss = self.test_task.inference()
        self.convergence_error[t] = loss
        self.test_accuracy[t] = acc
        print('global round : {} \t Loss: {.6f} \t Accuracy: {.3f}%'.format(t, loss, acc))
