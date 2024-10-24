import math
import time
import pickle
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models.resnet import BasicBlock
from Spiking_Models.activation import NoisySpike, InvSigmoid, InvRectangle
from Spiking_Models.neuron import LIFNeuron
from Spiking_Models.resnet import ResNet, SpikingBasicBlock, SpikingBottleneck, SmallResNet, ArtificialSmallResnet


class MultiStepNoisyRateScheduler:
    def __init__(self, init_p=1, reduce_ratio=0.9, milestones=[0.3, 0.7, 0.9, 0.95], num_epoch=100, start_epoch=0):
        self.reduce_ratio = reduce_ratio
        self.p = init_p
        self.milestones = [int(m * num_epoch) for m in milestones]
        self.num_epoch = num_epoch
        self.start_epoch = start_epoch

    def set_noisy_rate(self, p, model):
        for m in model.modules():
            if isinstance(m, NoisySpike):
                m.p = p

    def __call__(self, epoch, model):
        for one in self.milestones:
            if one + self.start_epoch == epoch:
                self.p *= self.reduce_ratio
                print('change noise rate as ' + str(self.p))
                self.set_noisy_rate(self.p, model)
                break


def TET_loss(outputs, labels, criterion, means, lamb):
    # print(outputs.shape)
    # print(labels.shape)
    T = outputs.size(0)
    loss_es = 0
    for t in range(T):
        loss_es += criterion(outputs[t], labels)
    loss_es = loss_es / T
    if lamb != 0:
        MMDLoss = nn.MSELoss()
        # mean_output = torch.mean(outputs, dim=1)
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
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        target = target.view(-1)
        if encoder is not None:
            data = encoder(data)

        if isinstance(optimizer, list):
            for optim in optimizer:
                optim.zero_grad()
        else:
            optimizer.zero_grad()

        output = model(data)
        # loss = TET_loss(output, target, evaluator, args.means, args.lamb)
        loss = evaluator(output, target)
        loss.backward()
        if isinstance(optimizer, list):
            for optim in optimizer:
                optim.step()
        else:
            optimizer.step()

        # predict = torch.argmax(output.mean(0), dim=1)
        predict = torch.argmax(output, dim=1)
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
            target = target.view(-1)
            # if encoder is not None:
            #     data = encoder(data)
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
                # loss = TET_loss(output, target, evaluator, args.means, args.lamb)
                loss = evaluator(output, target)
                # predict = torch.argmax(output.mean(0), dim=1)
                predict = torch.argmax(output, dim=1)
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


if __name__ == '__main__':
    from config import args

    num_class = 10
    dtype = torch.float
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open('../Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('../Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)

    train_loader = DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.num_workers)

    model_flag = 'snn'
    if model_flag == 'snn':
        decay = nn.Parameter(wrap_decay(args.decay))
        thresh = args.thresh
        args.alpha = 1 / args.alpha

        if args.act == 'mns_rec':
            inv_sg = InvRectangle(alpha=args.alpha, learnable=args.train_width, granularity=args.granularity)
        elif args.act == 'mns_sig':
            inv_sg = InvSigmoid(alpha=args.alpha, learnable=args.train_width)

        kwargs_spikes = {'nb_steps': args.T, 'vreset': 0, 'threshold': thresh,
                         'spike_fn': NoisySpike(p=args.p, inv_sg=inv_sg, spike=True), 'decay': decay}

        # model = ResNet(BasicBlock, [2, 2, 2, 2], num_class, bn_type=args.bn_type, **kwargs_spikes).to(device, dtype)
        model = SmallResNet(SpikingBasicBlock, [1, 2, 2, 2], num_class, bn_type=args.bn_type, **kwargs_spikes).to(
            device, dtype)
        params = split_params(model)
        spiking_params = [{'params': params[0], 'weight_decay': 0}]
        params = [{'params': params[1], 'weight_decay': args.wd}, {'params': params[2], 'weight_decay': 0}]

        if args.optim.lower() == 'sgdm':
            optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
        elif args.optim.lower() == 'adam':
            optimizer = optim.Adam(params, lr=args.lr, amsgrad=False)
        width_optim = optim.Adam(spiking_params, lr=args.width_lr)
        evaluator = torch.nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.num_epoch)
        rate_scheduler = MultiStepNoisyRateScheduler(init_p=args.p, reduce_ratio=args.gamma,
                                                     milestones=args.ns_milestone,
                                                     num_epoch=args.num_epoch, start_epoch=0)

        cur_lr = args.lr
        for epoch in tqdm(range(args.num_epoch)):
            # if args.optim.lower() == 'sgdm':
            #     optimizer = optim.SGD(params, lr=cur_lr, momentum=0.9)
            # elif args.optim.lower() == 'adam':
            #     optimizer = optim.Adam(params, lr=cur_lr, amsgrad=False)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
            # cur_lr = cur_lr * (1 + math.cos(math.pi * (epoch + 1) / args.num_epoch)) / (
            #             1 + math.cos(math.pi * epoch / args.num_epoch))

            cur_lr = args.lr * (1 + math.cos(math.pi * epoch / args.num_epoch)) / 2

            train_acc, train_loss = run_training(epoch, train_loader, [optimizer, width_optim], model, evaluator,
                                                 args=args)
            # scheduler.step()
            rate_scheduler(epoch, model)
            test_acc, test_loss = run_test(test_loader, model, evaluator, args=args)
            print(
                'Epoch {}: train loss {:.5f}, train acc {:.5f}, test loss {:.5f}, test acc {:.5f}'.format(epoch,
                                                                                                          train_loss,
                                                                                                          train_acc,
                                                                                                          test_loss,
                                                                                                          test_acc))
    elif model_flag == 'ann':
        model = ArtificialSmallResnet(BasicBlock, [1, 2, 2, 2], num_class).to(device, dtype)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        evaluator = torch.nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.num_epoch)
        cur_lr = args.lr
        for epoch in tqdm(range(args.num_epoch)):
            optimizer = optim.SGD(model.parameters(), lr=cur_lr, momentum=0.9)
            cur_lr = cur_lr * (1 + math.cos(math.pi * (epoch + 1) / args.num_epoch)) / (
                    1 + math.cos(math.pi * epoch / args.num_epoch))


            train_acc, train_loss = run_training(epoch, train_loader, optimizer, model, evaluator,
                                                 args=args)
            # scheduler.step()
            test_acc, test_loss = run_test(test_loader, model, evaluator, args=args)
            print(
                'Epoch {}: train loss {:.5f}, train acc {:.5f}, test loss {:.5f}, test acc {:.5f}'.format(epoch,
                                                                                                          train_loss,
                                                                                                          train_acc,
                                                                                                          test_loss,
                                                                                                          test_acc))
