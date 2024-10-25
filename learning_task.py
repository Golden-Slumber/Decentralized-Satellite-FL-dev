import os
import random
import sys

import math
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
from Spiking_Models.activation import NoisySpike, InvSigmoid, InvRectangle
from Spiking_Models.neuron import LIFNeuron
from Spiking_Models.resnet import SpikingBasicBlock, SmallResNet, ArtificialSmallResnet

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
        self.dataset = dataset
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        self.training_loss = 0.0

        self.model = EuroSatCNN().to(self.device)
        self.model_update(model)

    def local_training(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        self.model.train()
        for local_iter in range(self.args.local_iters):
            batch_count = 0
            self.training_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(-1)

                self.model.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_count += 1
                self.training_loss += loss.item()

            self.training_loss /= batch_count
            if self.args.verbose:
                print('\t Local Epoch : {} \t Loss: {:.6f}'.format(local_iter, self.training_loss))

    def inference(self):
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            batch_count = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(-1)

                outputs = self.model(images)
                batch_loss = self.criterion(outputs, labels)

                batch_count += 1
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
            loss /= batch_count

        accuracy = correct / total
        return accuracy, loss

    def get_model(self):
        return self.model.state_dict()

    def get_training_loss(self):
        _, loss = self.inference()
        self.training_loss = loss
        return self.training_loss

    def model_update(self, model_parameters):
        self.model.load_state_dict(model_parameters)


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


class EuroSatSNNTask(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.dataset = dataset
        self.dtype = torch.float
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        decay = nn.Parameter(wrap_decay(self.args.decay))
        thresh = self.args.thresh
        alpha = 1 / self.args.alpha
        if self.args.act == 'mns_rec':
            inv_sg = InvRectangle(alpha=alpha, learnable=self.args.train_width, granularity=self.args.granularity)
        elif self.args.act == 'mns_sig':
            inv_sg = InvSigmoid(alpha=alpha, learnable=self.args.train_width)
        kwargs_spikes = {'nb_steps': args.T, 'vreset': 0, 'threshold': thresh,
                         'spike_fn': NoisySpike(p=self.args.p, inv_sg=inv_sg, spike=True), 'decay': decay}

        self.model = SmallResNet(SpikingBasicBlock, [1, 2, 2, 2], num_classes=10, bn_type=self.args.bn_type,
                                 **kwargs_spikes).to(self.device, self.dtype)
        self.model_update(model)

        self.epoch = 0
        self.lr = self.args.lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = DataLoader(self.dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                      pin_memory=True, num_workers=self.args.num_workers)

        params = split_params(self.model)
        spiking_params = [{'params': params[0], 'weight_decay': 0}]
        params = [{'params': params[1], 'weight_decay': self.args.wd}, {'params': params[2], 'weight_decay': 0}]
        self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        self.width_optim = optim.Adam(spiking_params, lr=self.args.width_lr)

        self.local_train_loss = 0.0
        self.local_train_acc = 0.0

    def local_training(self):
        self.lr = self.args.lr * (1 + math.cos(math.pi * self.epoch / self.args.num_epoch)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        self.model.train()
        for local_iter in range(self.args.local_iters):
            loss_tot = []
            predict_tot = []
            label_tot = []

            for idx, (data, target) in enumerate(self.data_loader):
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
            print('\t Epoch [{}/{}], Local Iter [{}/{}] Local Loss: {:.5f}, Local Acc: {:.5f}'.format(self.epoch + 1,
                                                                                                      self.args.num_epoch,
                                                                                                      local_iter + 1,
                                                                                                      self.args.local_iters,
                                                                                                      local_train_loss,
                                                                                                      local_train_acc))

    def local_test(self):
        self.model.eval()
        with torch.no_grad():
            predict_tot = []
            label_tot = []
            loss_tot = []

            for idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1)
                output = self.model(data)

                loss = self.criterion(output, target)
                predict = torch.argmax(output, dim=1)
                predict_tot.append(predict)
                loss_tot.append(loss)
                label_tot.append(target)

            label_tot = torch.cat(label_tot)
            local_test_loss = torch.tensor(loss_tot).sum() / len(label_tot)
            predict_tot = torch.cat(predict_tot)
            local_test_acc = torch.mean((predict_tot == label_tot).float())

            return local_test_loss, local_test_acc

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                      pin_memory=True, num_workers=self.args.num_workers)

    def get_model(self):
        return self.model.state_dict()

    def get_training_stats(self):
        return self.local_train_loss, self.local_train_acc

    def model_update(self, model_parameters):
        self.model.load_state_dict(model_parameters)
