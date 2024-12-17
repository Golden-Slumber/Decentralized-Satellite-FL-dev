import sys
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter
import numpy
from STK_simulator.aggregation_routing_tree_construction import *
from Spiking_Models.layer import LIFLayer
from revised_satellite_system import *
from tqdm import tqdm
from Spiking_Models.spiking_learning import run_training, run_test

home_dir = './'
sys.path.append(home_dir)


def inference(device, data_loader, model, criterion):
    model.eval()
    spike_rates = {}
    batch_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LIFLayer):
            spike_rates[name] = 0

    with torch.no_grad():
        predict_tot = []
        label_tot = []
        loss_tot = []

        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            target = target.view(-1)
            output = model(data)

            batch_count += 1
            for name, module in model.named_modules():
                if isinstance(module, LIFLayer):
                    spike_rates[name] += module.avg_spike_rate
                    print('batch ' + str(idx) + ' module ' + name + ' avg_spike_rate ' + str(module.avg_spike_rate))

            loss = criterion(output, target)
            predict = torch.argmax(output, dim=1)
            predict_tot.append(predict)
            loss_tot.append(loss)
            label_tot.append(target)

        label_tot = torch.cat(label_tot)
        test_loss = torch.tensor(loss_tot).sum() / len(label_tot)
        predict_tot = torch.cat(predict_tot)
        test_acc = torch.mean((predict_tot == label_tot).float())

        for key in spike_rates.keys():
            spike_rates[key] /= batch_count

        # del predict_tot
        # del label_tot
        # del loss_tot
        # gc.collect()
        # torch.cuda.empty_cache()

        print(spike_rates)
        # return test_loss.item(), test_acc.item()
        return test_loss.detach_().item(), test_acc.detach_().item()


def plot_spiking_rates(spiking_rates):
    num_layers = len(spiking_rates)
    index = numpy.arange(num_layers)
    x_ticks = []
    values = []
    for key in spiking_rates.keys():
        x_ticks.append(key)
        values.append(spiking_rates[key])

    fig = plt.figure(figsize=(8, 10))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    bars = plt.bar(index, values, width=0.25, tick_label=x_ticks, color='#0072BD')
    for i in range(len(bars)):
        yval = values[i]
        plt.text(bars[i].get_x() + bars[i].get_width() / 2, yval, yval, ha='center', va='bottom', fontsize=20, color='#0072BD', weight='bold')
    plt.xlabel('Layers', fontsize=25)
    plt.ylabel('Spiking Rate', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()

    image_name = home_dir + 'Outputs/EuroSat_Spiking_Rate_CNN.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

def plot_energy_comparison(operations, spiking_rates, EMAC, EADD):
    num_layers = len(spiking_rates)
    index = numpy.arange(num_layers)
    x_ticks = []
    energy_ann = []
    energy_snn = []
    for key in operations.keys():
        x_ticks.append(key)
        energy_ann.append(operations[key] * EMAC / 1e6)
        energy_snn.append(operations[key] * spiking_rates[key] * EADD / 1e6)


    fig = plt.figure(figsize=(8, 10))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    bar_width = 0.25
    bars1 = plt.bar(index, energy_ann, width=bar_width, label='ANN', color='#D95319')
    bars2 = plt.bar(index+bar_width, energy_snn, width=bar_width, label='SNN', color='#EDB120')

    for i in range(len(bars1)):
        yval = round(energy_ann[i], 3)
        plt.text(bars1[i].get_x() + bars1[i].get_width() / 2, yval, yval, ha='center', va='bottom', fontsize=20, color='#D95319', weight='bold')
    for i in range(len(bars2)):
        yval = round(energy_snn[i], 3)
        plt.text(bars2[i].get_x() + bars2[i].get_width() / 2, yval, yval, ha='center', va='bottom', fontsize=20, color='#EDB120', weight='bold')

    plt.xlabel('Layers', fontsize=25)
    plt.ylabel(r'Energy Consumed ($\mu J$)', fontsize=25)
    plt.xticks(index+bar_width/2, x_ticks, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=25)
    plt.tight_layout()

    image_name = home_dir + 'Outputs/EuroSat_Energy_Comparison_CNN.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

def spiking_training(model_type):
    from config import args

    num_class = 10
    dtype = torch.float
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)

    train_loader = DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.num_workers)

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
    if model_type == 'resnet':
        model = SmallResNet(SpikingBasicBlock, [1, 2, 2, 2], num_class, bn_type=args.bn_type, **kwargs_spikes).to(
            device, dtype)
    elif model_type == 'cnn':
        model = SpikingCNN(num_classes=num_class, bn_type=args.bn_type, **kwargs_spikes).to(device, dtype)

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
    # rate_scheduler = MultiStepNoisyRateScheduler(init_p=args.p, reduce_ratio=args.gamma,
    #                                              milestones=args.ns_milestone,
    #                                              num_epoch=args.num_epoch, start_epoch=0)

    # cur_lr = args.lr
    # lr = args.lr * (args.lr_decay_ratio ** int((epoch + 1) / self.args.lr_decay_phase))
    for epoch in tqdm(range(args.num_epoch)):
        # if args.optim.lower() == 'sgdm':
        #     optimizer = optim.SGD(params, lr=cur_lr, momentum=0.9)
        # elif args.optim.lower() == 'adam':
        #     optimizer = optim.Adam(params, lr=cur_lr, amsgrad=False)
        cur_lr = args.lr * (args.lr_decay_ratio ** int((epoch + 1) / args.lr_decay_phase))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        # cur_lr = cur_lr * (1 + math.cos(math.pi * (epoch + 1) / args.num_epoch)) / (
        #             1 + math.cos(math.pi * epoch / args.num_epoch))

        # cur_lr = args.lr * (1 + math.cos(math.pi * epoch / args.num_epoch)) / 2

        train_acc, train_loss = run_training(epoch, train_loader, [optimizer, width_optim], model, evaluator,
                                             args=args)
        # scheduler.step()
        # rate_scheduler(epoch, model)
        test_acc, test_loss = run_test(test_loader, model, evaluator, args=args)
        print(
            'Epoch {}: train loss {:.5f}, train acc {:.5f}, test loss {:.5f}, test acc {:.5f}'.format(epoch,
                                                                                                      train_loss,
                                                                                                      train_acc,
                                                                                                      test_loss,
                                                                                                      test_acc))
    torch.save(model.state_dict(), './Outputs/pre_trained_spiking_' + str(model_type) + '.pt')


if __name__ == '__main__':
    model_type = 'cnn'
    train_flag = False

    spiking_rates = {'conv1.2': 0.4923, 'conv2.2': 0.2083, 'fc.1': 0.9794}
    plot_spiking_rates(spiking_rates)

    operations = {'conv1.2': 3 * 3 * 3 * 4 * 64 * 64, 'conv2.2': 3 * 3 * 4 * 8 * 32 * 32, 'fc.1': 8 * 16 * 16 * 32 }
    EMAC = 4.6
    EADD = 0.9
    plot_energy_comparison(operations, spiking_rates, EMAC, EADD)

    # if train_flag:
    #     spiking_training(model_type)
    # else:
    #     from config import args
    #
    #     num_class = 10
    #     dtype = torch.float
    #     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #
    #     with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
    #         train_set = pickle.load(f)
    #     with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
    #         test_set = pickle.load(f)
    #
    #     train_loader = DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True, pin_memory=True,
    #                               num_workers=args.num_workers)
    #     test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False, pin_memory=True,
    #                              num_workers=args.num_workers)
    #
    #     decay = nn.Parameter(wrap_decay(args.decay))
    #     thresh = args.thresh
    #     args.alpha = 1 / args.alpha
    #
    #     if args.act == 'mns_rec':
    #         inv_sg = InvRectangle(alpha=args.alpha, learnable=args.train_width, granularity=args.granularity)
    #     elif args.act == 'mns_sig':
    #         inv_sg = InvSigmoid(alpha=args.alpha, learnable=args.train_width)
    #
    #     kwargs_spikes = {'nb_steps': args.T, 'vreset': 0, 'threshold': thresh,
    #                      'spike_fn': NoisySpike(p=args.p, inv_sg=inv_sg, spike=True), 'decay': decay}
    #
    #     if model_type == 'resnet':
    #         model = SmallResNet(SpikingBasicBlock, [1, 2, 2, 2], num_class, bn_type=args.bn_type, **kwargs_spikes).to(
    #             device, dtype)
    #     elif model_type == 'cnn':
    #         model = SpikingCNN(num_classes=num_class, bn_type=args.bn_type, **kwargs_spikes).to(device, dtype)
    #     model_state_dict = torch.load('./Outputs/pre_trained_spiking_' + str(model_type) + '.pt')
    #     model(train_set[0][0].unsqueeze(0).to(device, dtype))
    #     model.load_state_dict(model_state_dict)
    #     model.eval()
    #     with torch.no_grad():
    #         model(train_set[0][0].unsqueeze(0).to(device, dtype))
    #         for name, module in model.named_modules():
    #             if isinstance(module, LIFLayer):
    #                 print(name)
    #                 print(module.avg_spike_rate)
    #     evaluator = torch.nn.CrossEntropyLoss()
    #     inference(device, test_loader, model, evaluator)
