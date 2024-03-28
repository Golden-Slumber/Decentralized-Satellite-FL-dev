import os
import random
import sys
import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from sklearn.model_selection import train_test_split
import pickle

home_dir = './'
sys.path.append(home_dir)


class EuroSatCNN(nn.Module):
    def __init__(self):
        super(EuroSatCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=8 * 16 * 16, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = out.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class EuroSatTask(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.dataset = deepcopy(dataset)
        self.model = deepcopy(model)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)

    def local_training(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        self.model.train()
        for local_iter in range(self.args.local_iters):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(-1)

                self.model.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(local_iter,
                                                                                        batch_idx * len(images),
                                                                                        len(self.train_loader.dataset),
                                                                                        100. * batch_idx / len(
                                                                                            self.train_loader),
                                                                                        loss.item()))

    def inference(self):
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(-1)

                outputs = self.model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct = torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct / total
        return accuracy, loss

    def get_model(self):
        return self.model.state_dict()

    def model_update(self, model_parameters):
        self.model.load_state_dict(model_parameters)
