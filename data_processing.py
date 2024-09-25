import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy


class DVSAugmentedDataset(Dataset):
    def __init__(self, dataset, train=True, transform=None, target_transform=None):
        self.dataset = dataset
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, item):
    #     data, target = self.dataset[item]
    #     new_data = []
    #     for t in range(data.se)