import sys
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter
import numpy
from STK_simulator.aggregation_routing_tree_construction import *
from revised_satellite_system import *

home_dir = './'
sys.path.append(home_dir)

def plot_results(acc, loss, iterations, legends, scheme):
    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    epoch_list = list(range(iterations))
    tick_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for i in range(len(legends)):
        line, = plt.plot(epoch_list, numpy.median(loss[i], axis=0), color=color_list[i], linestyle='-',
                         marker=marker_list[i],
                         markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, markevery=5)
        line_list.append(line)
    plt.legend(line_list, legends, fontsize=25)
    plt.xticks(tick_list, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Training Epochs', fontsize=25)
    plt.ylabel('Training Loss', fontsize=25)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid()

    image_name = home_dir + 'Outputs/EuroSatSNN_Comparison_demo_' + scheme + '_accuracy.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line_list = []
    epoch_list = list(range(iterations))
    tick_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
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

    repeat = 1
    num_planes = 9
    num_satellites = 10
    satellites_by_plane = [num_satellites for i in range(num_planes)]
    legends = ['RelaySum (Proposed Inter-Plane Aggregation)', 'Gossip (Naive Inter-Plane Aggregation)',
               'All-Reduce (Baseline)']
    scheme = 'Inter-Plane Comparison'

    constellation = ConstellationLearning(num_planes, satellites_by_plane, train_set, test_set, args)
    constellation.dataset_partition()

    acc = numpy.zeros((3, repeat, args.num_epoch))
    loss = numpy.zeros((3, repeat, args.num_epoch))
    saved_acc = numpy.zeros((3, args.num_epoch))
    saved_loss = numpy.zeros((3, args.num_epoch))



    for r in range(repeat):
        connectivity_matrix = WalkerStarConnectivity
        n = len(connectivity_matrix)
        for i in range(n):
            for j in range(n):
                if i != j and connectivity_matrix[i][j] == 0.0:
                    connectivity_matrix[i][j] = -1
        aggregation_matrix = MDST_construction(connectivity_matrix)
        print(aggregation_matrix)

        print('RelaySum Training')
        constellation.spike_learning_initialization()
        constellation.inter_plane_aggregation_configuration(aggregation_matrix, RELAYSUM)

        for epoch in range(args.num_epoch):
            constellation.constellation_learning(epoch)
            acc[0, r, epoch] = constellation.test_accuracy[epoch]
            loss[0, r, epoch] = constellation.convergence_error[epoch]
            saved_acc[0, epoch] = constellation.test_accuracy[epoch]
            saved_loss[0, epoch] = constellation.convergence_error[epoch]

        print('Gossip Training')
        constellation.spike_learning_initialization()
        constellation.inter_plane_aggregation_configuration(aggregation_matrix, GOSSIP)
        for epoch in range(args.num_epoch):
            constellation.constellation_learning(epoch)
            acc[1, r, epoch] = constellation.test_accuracy[epoch]
            loss[1, r, epoch] = constellation.convergence_error[epoch]
            saved_acc[1, epoch] = constellation.test_accuracy[epoch]
            saved_loss[1, epoch] = constellation.convergence_error[epoch]

        print('All Reduce Training')
        constellation.spike_learning_initialization()
        constellation.inter_plane_aggregation_configuration(aggregation_matrix, ALLREDUCE)
        for epoch in range(args.num_epoch):
            constellation.constellation_learning(epoch)
            acc[2, r, epoch] = constellation.test_accuracy[epoch]
            loss[2, r, epoch] = constellation.convergence_error[epoch]
            saved_acc[2, epoch] = constellation.test_accuracy[epoch]
            saved_loss[2, epoch] = constellation.convergence_error[epoch]

        out_file_name = home_dir + 'Outputs/EuroSat_SNN_InterPlane_Comparison_Repeat_' + str(
            r) + '_results.npz'
        numpy.savez(out_file_name, acc=saved_acc, loss=saved_loss)
    plot_results(acc, loss, args.num_epoch, legends, scheme)
