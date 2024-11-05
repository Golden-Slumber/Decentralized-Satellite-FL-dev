import sys
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter
import numpy
from satellite_system import *
from learning_task import *
from utils import *
from Spiking_Models.activation import NoisySpike, InvSigmoid, InvRectangle
from Spiking_Models.neuron import LIFNeuron
from Spiking_Models.resnet import SpikingBasicBlock, SmallResNet, ArtificialSmallResnet
from STK_simulator.constellation_config import WalkerStarConnectivity
from STK_simulator.aggregation_routing_tree_construction import *

home_dir = './'
sys.path.append(home_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def plot_results(acc, loss, iterations, legends, scheme):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    epoch_list = list(range(iterations))
    tick_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    for i in range(len(legends)):
        line, = plt.plot(epoch_list, numpy.median(acc[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=5)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=25)
    plt.xticks(tick_list, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Training Epochs', fontsize=25)
    plt.ylabel('Test Accuracy', fontsize=25)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/EuroSatSNN_Comparison_demo_' + scheme + '_accuracy.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    from config import args

    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)

    num_planes = 9
    num_satellites = 10
    datasize = int(len(train_set) / num_planes / num_satellites)
    datasize_array_by_plane = [datasize * numpy.ones(num_satellites).astype(int) for i in range(num_planes)]
    satellites_by_plane = [num_satellites for i in range(num_planes)]

    decay = nn.Parameter(wrap_decay(args.decay))
    thresh = args.thresh
    alpha = 1 / args.alpha
    if args.act == 'mns_rec':
        inv_sg = InvRectangle(alpha=alpha, learnable=args.train_width, granularity=args.granularity)
    elif args.act == 'mns_sig':
        inv_sg = InvSigmoid(alpha=alpha, learnable=args.train_width)
    kwargs_spikes = {'nb_steps': args.T, 'vreset': 0, 'threshold': thresh,
                     'spike_fn': NoisySpike(p=args.p, inv_sg=inv_sg, spike=True), 'decay': decay}

    init_model = SmallResNet(SpikingBasicBlock, [1, 2, 2, 2], num_classes=10, bn_type=args.bn_type,
                             **kwargs_spikes).to(device, dtype).state_dict()
    for key in init_model.keys():
        print(key)

    constellation = Constellation(num_planes, satellites_by_plane, train_set, test_set, datasize_array_by_plane,
                                  init_model, args)
    print('constellation initialization')

    repeat = 1
    legends = ['RelaySum', 'Gossip', 'All-Reduce']
    acc = numpy.zeros((3, repeat, args.num_epoch))
    loss = numpy.zeros((3, repeat, args.num_epoch))
    saved_acc = numpy.zeros((3, args.num_epoch))
    saved_loss = numpy.zeros((3, args.num_epoch))

    for r in range(repeat):
        print('RelaySum Training')
        for p in constellation.plane_list:
            p.relay_sum_initialization(num_planes)
        constellation.reset_constellation(init_model)

        connectivity_matrix = WalkerStarConnectivity
        n = len(connectivity_matrix)
        for i in range(n):
            for j in range(n):
                if i != j and connectivity_matrix[i][j] == 0.0:
                    connectivity_matrix[i][j] = -1
        # _, routing_matrix = simplified_MDST_construction(connectivity_matrix)
        # average_matrix = numpy.zeros((num_planes, num_planes))
        # for p in range(num_planes):
        #     average_matrix[p, p] = 1
        #     for q in range(num_planes):
        #         if routing_matrix[p, q] == q + 1:
        #             average_matrix[p, q] = 1
        average_matrix = MDST_construction(connectivity_matrix)
        print(average_matrix)
        constellation.set_connectivity_matrix(average_matrix)
        for t in range(args.num_epoch):
            constellation.constellation_training(RELAYSUM)
            constellation.save_metric_v3(t)
        for t in range(args.num_epoch):
            acc[0, r, t] = constellation.test_accuracy[t]
            loss[0, r, t] = constellation.convergence_error[t]
            saved_acc[0, t] = constellation.test_accuracy[t]
            saved_loss[0, t] = constellation.convergence_error[t]

        constellation.reset_constellation(init_model)
        print('Gossip Training')
        # constellation.set_connectivity_matrix(connectivity_matrix)
        for t in range(args.num_epoch):
            # connectivity_matrix = fixed_binary_tree_topology(num_planes)
            constellation.constellation_training(GOSSIP)
            constellation.save_metric_v3(t)
        for t in range(args.num_epoch):
            acc[1, r, t] = constellation.test_accuracy[t]
            loss[1, r, t] = constellation.convergence_error[t]
            saved_acc[1, t] = constellation.test_accuracy[t]
            saved_loss[1, t] = constellation.convergence_error[t]

        constellation.reset_constellation(init_model)
        print('AllReduce Training')
        for t in range(args.num_epoch):
            # connectivity_matrix = fixed_binary_tree_topology(num_planes)
            # constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(ALLREDUCE)
            constellation.save_metric_v3(t)
        for t in range(args.num_epoch):
            acc[2, r, t] = constellation.test_accuracy[t]
            loss[2, r, t] = constellation.convergence_error[t]
            saved_acc[2, t] = constellation.test_accuracy[t]
            saved_loss[2, t] = constellation.convergence_error[t]
        out_file_name = home_dir + 'Outputs/EuroSatSNN_Comparison__Repeat_' + str(
            r) + '_results.npz'
        numpy.savez(out_file_name, acc=saved_acc, loss=saved_loss)
    plot_results(acc, loss, args.iterations, legends, 'WalkerStar')