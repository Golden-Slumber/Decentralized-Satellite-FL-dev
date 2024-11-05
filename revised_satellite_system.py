import sys
import numpy
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from utils import Dirichlet_non_iid_distribution
from constants import *


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class ConstellationLearning(object):
    def __init__(self, num_planes, satellites_by_planes, train_dataset, test_dataset, init_model, args):
        # satellite constellation configuration
        self.num_planes = num_planes
        self.satellites_by_planes = satellites_by_planes
        self.connectivity_matrix = None

        # learning parameters
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.global_model = deepcopy(init_model)
        self.local_models = [[] for i in range(self.num_planes)]
        self.local_dataset_idxs = [[] for i in range(self.num_planes)]

        # performance metrics
        self.convergence_error = numpy.zeros(self.args.num_epoch)
        self.consensus_error = numpy.zeros(self.args.num_epoch)
        self.test_accuracy = numpy.zeros(self.args.num_epoch)

    def dataset_partition(self):
        