import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from Spiking_Models.activation import NoisySpike, InvSigmoid, InvRectangle
from Spiking_Models.neuron import LIFNeuron
from Spiking_Models.resnet import ResNet


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(0)
    loss_es = 0
    for t in range(T):
        loss_es += criterion(outputs[t, ...], labels)
    loss_es = loss_es / T
    if lamb != 0:
        MMDLoss = nn.MSELoss
        y = torch.zeros_like(outputs).fill_(means)
        loss_mmd = MMDLoss(outputs, y)
    else:
        loss_mmd = 0
    return (1 - lamb) * loss_es + lamb * loss_mmd


def run_training(epoch, train_loader, optimizer, model, evaluator, args=None, encoder=None):
    loss_record = []
    predict_tot = []
    label_tot = []

    model.train()
    strat_time = time.time()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        if encoder is not None:
            data = encoder(data)

        optimizer.zero_grad()
        if isinstance(optimizer, list):
            for optim in optimizer:
                optim.zero_grad()

        output = model(data)
        loss = TET_loss(output, target, evaluator, args.means, args.lamb)
        loss.backward()
        if isinstance(optimizer, list):
            for optim in optimizer:
                optim.zero_grad()
        else:
            optimizer.step()

        predict = torch.argmax(output.mean(0), dim=1)
        loss_record.append(loss.detach().cpu())
        predict_tot.append(predict)
        label_tot.append(target)

        if (idx + 1) % args.log_interval == 0:
            print(
                '\t Epoch [{}/{}], Step [{}/{}] Loss: {:.6f}'.format(epoch, args.num_epoch, idx + 1,
                                                                     len(train_loader.dataset) // args.train_batch_size,
                                                                     loss_record[-1] / args.train_batch_size))

    predict_tot = torch.cat(predict_tot)
    label_tot = torch.cat(label_tot)
    train_acc = torch.mean((predict_tot == label_tot).float())
    train_loss = torch.tensor(loss_record).sum() / len(label_tot)
    return train_acc, train_loss


def run_test(test_loader, model, evaluator, args=None, encoder=None):
    model.eval()
    with torch.no_grad():
        predict_tot = {}
        label_tot = []
        loss_record = []
        key = 'ann' if encoder is None else 'snn'

        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            if encoder is not None:
                data = encoder(data)
            output = model(data)
            if isinstance(output, dict):
                for t in output.keys():
                    if t not in predict_tot.keys():
                        predict_tot[t] = []
                    predict = torch.argmax(output[t], dim=1)
                    predict_tot[t].append(predict)
                loss = evaluator(output[encoder.nb_steps], target)
            else:
                if key not in predict_tot.keys():
                    predict_tot[key] = []
                loss = TET_loss(output, target, evaluator, args.means, args.lamb)
                predict = torch.argmax(output.mean(0), dim=1)
                predict_tot[key].append(predict)
            loss_record.append(loss)
            label_tot.append(target)

        label_tot = torch.cat(label_tot)
        test_loss = torch.tensor(loss_record).sum() / len(label_tot)
        if 'ann' not in predict_tot.keys() and 'snn' not in predict_tot.keys():
            test_acc = {}
            for t in predict_tot.keys():
                test_acc[t] = torch.mean((torch.cat(predict_tot[t]) == label_tot).float())
        else:
            predict_tot = torch.cat(predict_tot[key])
            test_acc = torch.mean((predict_tot == label_tot).float())
        return test_acc, test_loss


