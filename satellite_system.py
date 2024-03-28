import os
import random
import sys
import numpy
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
from learning_task import EuroSatTask

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
        self.init_model = deepcopy(init_model)
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

    def get_satellites(self):
        return self.satellite_list

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
                intra_plane_model[key] += local_models[i] * self.sum_weight_arr[i]
        return intra_plane_model


class Constellation(object):
    def __init__(self, num_planes, satellites_by_plane, constellation_dataset, datasize_by_plane, init_model, args):
        self.num_planes = num_planes
        self.satellites_by_plane = deepcopy(satellites_by_plane)
        self.constellation_dataset = deepcopy(constellation_dataset)
        self.datasize_by_plane = deepcopy(datasize_by_plane)
        self.sum_weight_arr = numpy.array(list(map(sum, datasize_by_plane))) / sum(map(sum, datasize_by_plane))
        self.args = args

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

