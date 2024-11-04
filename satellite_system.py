import os
import random
import sys
import numpy
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from learning_task import EuroSatTask, EuroSatSNNTask
from utils import Dirichlet_non_iid_distribution
from constants import *

home_dir = './'
sys.path.append(home_dir)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalDataset(Dataset):
    def __init__(self, data, label):
        self.data = deepcopy(data)
        self.label = deepcopy(label)
        self.len = data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.data[item], self.label[item]


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
        print('Plane {} begin initialization'.format(plane_idx))
        self.plane_idx = plane_idx
        self.num_satellites = num_satellites
        self.plane_dataset = plane_dataset
        self.datasize_by_satellite = deepcopy(datasize_by_satellite)
        self.sum_weight_arr = deepcopy(datasize_by_satellite) / numpy.sum(datasize_by_satellite)
        self.intra_plane_model = deepcopy(init_model)
        self.args = args

        self.satellite_list = []
        generator = torch.Generator().manual_seed(42)
        self.satellite_dataset_list = random_split(plane_dataset, datasize_by_satellite, generator=generator)
        # total_sample_idxs = [i for i in range(len(plane_dataset))]
        for satellite_idx in range(num_satellites):
            # local_sample_idx = set(
            #     numpy.random.choice(total_sample_idxs, int(datasize_by_satellite[satellite_idx]), replace=False))
            # total_sample_idxs = list(set(total_sample_idxs) - local_sample_idx)

            # local_data, local_label = plane_dataset[list(local_sample_idx)]
            # new_learning_task = EuroSatTask(args, self.satellite_dataset_list[satellite_idx], init_model)
            new_learning_task = EuroSatSNNTask(args, self.satellite_dataset_list[satellite_idx], init_model)
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
            self.received_relay_models.append(deepcopy(zero_model))
            self.transmitted_relay_models.append(deepcopy(zero_model))
            self.received_relay_counts.append(0)
            self.transmitted_relay_counts.append(0)

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
        self.constellation_dataset = constellation_dataset
        self.test_dataset = test_dataset
        self.datasize_by_plane = deepcopy(datasize_by_plane)
        self.sum_weight_arr = numpy.array(list(map(sum, datasize_by_plane))) / sum(map(sum, datasize_by_plane))
        self.args = args
        self.connectivity_matrix = None
        self.global_model = deepcopy(init_model)

        # performance metrics
        self.convergence_error = numpy.zeros(self.args.num_epoch)
        self.consensus_error = numpy.zeros(self.args.num_epoch)
        self.test_accuracy = numpy.zeros(self.args.num_epoch)
        # self.test_task = EuroSatTask(args, test_dataset, init_model)
        # self.training_task = EuroSatTask(args, constellation_dataset, init_model)
        self.learning_task = EuroSatSNNTask(args, constellation_dataset, init_model)

        self.plane_list = []
        # generator = torch.Generator().manual_seed(42)
        # self.plane_dataset_list = random_split(constellation_dataset, list(map(sum, datasize_by_plane)),
        #                                        generator=generator)
        # tmp_loader = DataLoader(constellation_dataset, batch_size=len(constellation_dataset), shuffle=False)
        targets = []
        for i in range(len(constellation_dataset)):
            _, target = constellation_dataset[i]
            targets.append(target.item())
        indices_per_plane = Dirichlet_non_iid_distribution(targets, self.args.plane_alpha, num_planes,
                                                           n_auxi_devices=10, seed=0)
        indices_per_plane = numpy.array_split(numpy.concatenate(indices_per_plane), num_planes)
        self.plane_dataset_list = [Subset(constellation_dataset, indices) for indices in indices_per_plane]
        # total_sample_idxs = [i for i in range(len(constellation_dataset))]
        for plane_idx in range(num_planes):
            # print(len(self.plane_dataset_list[plane_idx]))
            # plane_sample_idx = set(
            #     numpy.random.choice(total_sample_idxs, int(sum(datasize_by_plane[plane_idx])), replace=False))
            # total_sample_idxs = list(set(total_sample_idxs) - plane_sample_idx)

            # local_data, local_label = constellation_dataset[list(plane_sample_idx)]
            new_plane = Plane(plane_idx, self.satellites_by_plane[plane_idx],
                              self.plane_dataset_list[plane_idx], datasize_by_plane[plane_idx],
                              init_model, args)
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
            intra_plane_model_list.append(plane.get_intra_plane_model())

        if aggregation_scheme == GOSSIP:
            # generate mixing matrix
            # average_matrix = numpy.ones((self.num_planes, self.num_planes))
            # for p in range(self.num_planes):
            #     num_neighbors = numpy.sum(self.connectivity_matrix[p])
            #     average_matrix[p] = self.connectivity_matrix[p] / num_neighbors
            average_matrix = numpy.zeros((self.num_planes, self.num_planes))
            for p in range(self.num_planes):
                if p == 0:
                    average_matrix[p] = 1 / 2
                    average_matrix[p + 1] = 1 / 2
                elif p == self.num_planes - 1:
                    average_matrix[p] = 1 / 2
                    average_matrix[p - 1] = 1 / 2
                else:
                    average_matrix[p] = 1 / 3
                    average_matrix[p - 1] = 1 / 3
                    average_matrix[p + 1] = 1 / 3

            # print(average_matrix)
            for p in range(self.num_planes):
                plane_model = model_aggregation(intra_plane_model_list, list(average_matrix[p]))
                self.plane_list[p].set_intra_plane_model(plane_model)
        elif aggregation_scheme == RELAYSUM:
            for p in range(self.num_planes):
                # find neighbors to relay messages in the previous round
                for neighbor in range(self.num_planes):
                    if self.connectivity_matrix[p, neighbor] == 1 and p != neighbor:
                        new_relay_model = deepcopy(intra_plane_model_list[p])
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
                    if self.connectivity_matrix[neighbor, p] == 1 and p != neighbor:
                        relay_count, relay_model = self.plane_list[neighbor].get_transmitted_message(p)
                        self.plane_list[p].set_received_message(neighbor, relay_count, relay_model)
                        plane_count += relay_count
                        for key in plane_model.keys():
                            plane_model[key] += relay_model[key]
                for key in plane_model.keys():
                    plane_model[key] /= plane_count
                self.plane_list[p].set_intra_plane_model(plane_model)
        elif aggregation_scheme == ALLREDUCE:
            weight_list = []
            for p in range(self.num_planes):
                weight_list.append(1 / self.num_planes)
            all_reduced_model = model_aggregation(intra_plane_model_list, weight_list)
            for p in range(self.num_planes):
                self.plane_list[p].set_intra_plane_model(all_reduced_model)

        self.global_model = deepcopy(self.plane_list[0].get_intra_plane_model())
        for key in self.global_model.keys():
            self.global_model[key] /= self.num_planes
            for i in range(1, self.num_planes):
                self.global_model[key] += deepcopy(self.plane_list[i].get_intra_plane_model())[key] / self.num_planes

    def save_metric_v3(self, t):
        loss = 0.0
        acc = 0.0
        self.learning_task.set_dataset(self.constellation_dataset)
        for p in range(self.num_planes):
            plane_model = self.plane_list[p].get_intra_plane_model()
            self.learning_task.model_update(plane_model)
            plane_loss, _ = self.learning_task.local_test()
            loss += plane_loss / self.num_planes
        self.learning_task.set_dataset(self.test_dataset)
        for p in range(self.num_planes):
            plane_model = self.plane_list[p].get_intra_plane_model()
            self.learning_task.model_update(plane_model)
            _, plane_acc = self.learning_task.local_test()
            acc += plane_acc / self.num_planes
        self.convergence_error[t] = loss
        self.test_accuracy[t] = acc
        print('global round : {} \t Loss: {:.6f} \t Accuracy: {:.3f}%'.format(t, loss, 100. * acc))

    def save_metric_v2(self, t):
        loss = 0.0
        acc = 0.0
        for p in range(self.num_planes):
            plane_model = self.plane_list[p].get_intra_plane_model()
            self.training_task.model_update(plane_model)
            self.test_task.model_update(plane_model)
            _, plane_loss = self.training_task.inference()
            plane_acc, _ = self.test_task.inference()
            loss += plane_loss / self.num_planes
            acc += plane_acc / self.num_planes
        self.convergence_error[t] = loss
        self.test_accuracy[t] = acc
        print('global round : {} \t Loss: {:.6f} \t Accuracy: {:.3f}%'.format(t, loss, 100. * acc))

    def save_metric(self, t):
        training_loss = 0.0
        for p in range(self.num_planes):
            satellite_count = 0
            plane_loss = 0.0
            for satellite in self.plane_list[p].get_satellites():
                satellite_loss = satellite.learning_task.get_training_loss()
                plane_loss += satellite_loss
                satellite_count += 1
            plane_loss /= satellite_count
            training_loss += plane_loss / self.num_planes
        self.test_task.model_update(self.global_model)
        acc, _ = self.test_task.inference()
        self.convergence_error[t] = training_loss
        self.test_accuracy[t] = acc
        print('global round : {} \t Loss: {:.6f} \t Accuracy: {:.3f}%'.format(t, training_loss, 100. * acc))

    def reset_constellation(self, ini_model):
        for p in self.plane_list:
            p.set_intra_plane_model(ini_model)
