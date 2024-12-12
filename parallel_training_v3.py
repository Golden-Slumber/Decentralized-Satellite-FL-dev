import sys
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter
import numpy
from STK_simulator.aggregation_routing_tree_construction import *
from revised_satellite_system import *
import gc

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

    image_name = home_dir + 'Outputs/EuroSatSNN_Comparison_demo_' + scheme + '_para_loss_v3.pdf'
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

    image_name = home_dir + 'Outputs/EuroSatSNN_Comparison_demo_' + scheme + '_para_accuracy_v3.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    from config import args4
    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)

    repeat = 1
    num_planes = 5
    num_satellites = 10
    satellites_by_plane = [num_satellites for i in range(num_planes)]
    legends = ['RelaySum (Proposed Inter-Plane Aggregation)', 'Gossip (Naive Inter-Plane Aggregation)',
               'All-Reduce (Baseline)']
    scheme = 'Inter-Plane Comparison'
    print('training epochs: ' + str(args4.num_epoch) + ' plane-alpha: ' + str(args4.plane_alpha))

    # print('dataset partition...')
    # # non-IID inter-plane partition
    # local_dataset_indices = [[[] for j in range(satellites_by_plane[i])] for i in range(num_planes)]
    # targets = []
    # for i in range(len(train_set)):
    #     _, target = train_set[i]
    #     targets.append(target.item())
    # indices_per_plane = Dirichlet_non_iid_distribution(targets, args4.plane_alpha, num_planes,
    #                                                    n_auxi_devices=3, seed=0)
    # for i in range(num_planes):
    #     num_samples_per_satellite = int(len(indices_per_plane[i]) / satellites_by_plane[i])
    #     plane_indices = deepcopy(indices_per_plane[i])
    #     for j in range(satellites_by_plane[i]):
    #         local_dataset_indices[i][j] = numpy.random.choice(plane_indices, num_samples_per_satellite,
    #                                                                replace=False)
    #         plane_indices = list(set(plane_indices) - set(local_dataset_indices[i][j]))
    #
    # with open(home_dir + 'Outputs/EuroSat_SNN_planes_' + str(num_planes) + '_satellites_' + str(
    #     num_satellites) + '_alpha_' + str(args4.plane_alpha) + '_auxi_' + str(3) + '_TrainSetPartition.pkl', 'wb') as f:
    #     pickle.dump(local_dataset_indices, f)

    with open(home_dir + 'Outputs/EuroSat_SNN_planes_' + str(num_planes) + '_satellites_' + str(
        num_satellites) + '_alpha_' + str(args4.plane_alpha) + '_auxi_' + str(3) + '_TrainSetPartition.pkl', 'rb') as f:
        local_dataset_indices = pickle.load(f)

    constellation = ConstellationLearning(num_planes, satellites_by_plane, train_set, test_set, local_dataset_indices, args4)
    # constellation.dataset_partition()

    acc = numpy.zeros((3, repeat, args4.num_epoch))
    loss = numpy.zeros((3, repeat, args4.num_epoch))
    training_loss = numpy.zeros((3, repeat, args4.num_epoch))
    test_loss = numpy.zeros((3, repeat, args4.num_epoch))
    saved_acc = numpy.zeros((3, args4.num_epoch))
    saved_loss = numpy.zeros((3, args4.num_epoch))
    saved_training_loss = numpy.zeros((3, args4.num_epoch))
    saved_test_loss = numpy.zeros((3, args4.num_epoch))

    for r in range(repeat):
        # connectivity_matrix = WalkerStarConnectivity
        # n = len(connectivity_matrix)
        # for i in range(n):
        #     for j in range(n):
        #         if i != j and connectivity_matrix[i][j] == 0.0:
        #             connectivity_matrix[i][j] = -1
        # aggregation_matrix = MDST_construction(connectivity_matrix)
        aggregation_matrix = numpy.zeros((num_planes, num_planes))
        for i in range(num_planes):
            for j in range(num_planes):
                if i == j or i - 1 == j or i + 1 == j:
                    aggregation_matrix[i][j] = 1
        print(aggregation_matrix)

        print('RelaySum Training')
        constellation.spike_learning_initialization()
        constellation.inter_plane_aggregation_configuration(aggregation_matrix, RELAYSUM)

        for epoch in range(args4.num_epoch):
            constellation.constellation_learning(epoch)
            acc[0, r, epoch] = constellation.test_accuracy[epoch]
            loss[0, r, epoch] = constellation.convergence_error[epoch]
            training_loss[0, r, epoch] = constellation.training_loss[epoch]
            test_loss[0, r, epoch] = constellation.test_loss[epoch]
            saved_acc[0, epoch] = constellation.test_accuracy[epoch]
            saved_loss[0, epoch] = constellation.convergence_error[epoch]
            saved_training_loss[0, epoch] = constellation.training_loss[epoch]
            saved_test_loss[0, epoch] = constellation.test_loss[epoch]
            # if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        out_file_name = home_dir + 'Outputs/EuroSat_SNN_InterPlane_Comparison_T_' + str(args4.T) + '_alpha_' + str(
            args4.plane_alpha) + '_num_planes_' + str(num_planes) + '_lr_' + str(args4.lr) + '_repeat_' + str(r) + '_RelaySum_para_results_v3.npz'
        numpy.savez(out_file_name, acc=saved_acc, loss=saved_loss, trainloss=saved_training_loss,
                    testloss=saved_test_loss)

        print('Gossip Training')
        constellation.spike_learning_initialization()
        constellation.inter_plane_aggregation_configuration(aggregation_matrix, GOSSIP)
        for epoch in range(args4.num_epoch):
            constellation.constellation_learning(epoch)
            acc[1, r, epoch] = constellation.test_accuracy[epoch]
            loss[1, r, epoch] = constellation.convergence_error[epoch]
            training_loss[1, r, epoch] = constellation.training_loss[epoch]
            test_loss[1, r, epoch] = constellation.test_loss[epoch]
            saved_acc[1, epoch] = constellation.test_accuracy[epoch]
            saved_loss[1, epoch] = constellation.convergence_error[epoch]
            saved_training_loss[1, epoch] = constellation.training_loss[epoch]
            saved_test_loss[1, epoch] = constellation.test_loss[epoch]
            # if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        out_file_name = home_dir + 'Outputs/EuroSat_SNN_InterPlane_Comparison_T_' + str(args4.T) + '_alpha_' + str(
            args4.plane_alpha) + '_num_planes_' + str(num_planes) + '_lr_' + str(args4.lr) + '_repeat_' + str(r) + '_Gossip_para_results_v3.npz'
        numpy.savez(out_file_name, acc=saved_acc, loss=saved_loss, trainloss=saved_training_loss,
                    testloss=saved_test_loss)

        print('All Reduce Training')
        constellation.spike_learning_initialization()
        constellation.inter_plane_aggregation_configuration(aggregation_matrix, ALLREDUCE)
        for epoch in range(args4.num_epoch):
            constellation.constellation_learning(epoch)
            acc[2, r, epoch] = constellation.test_accuracy[epoch]
            loss[2, r, epoch] = constellation.convergence_error[epoch]
            training_loss[2, r, epoch] = constellation.training_loss[epoch]
            test_loss[2, r, epoch] = constellation.test_loss[epoch]
            saved_acc[2, epoch] = constellation.test_accuracy[epoch]
            saved_loss[2, epoch] = constellation.convergence_error[epoch]
            saved_training_loss[2, epoch] = constellation.training_loss[epoch]
            saved_test_loss[2, epoch] = constellation.test_loss[epoch]
            # if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        out_file_name = home_dir + 'Outputs/EuroSat_SNN_InterPlane_Comparison_T_' + str(args4.T) + '_alpha_' + str(
            args4.plane_alpha) + '_num_planes_' + str(num_planes) + '_lr_' + str(args4.lr) + '_repeat_' + str(r) + '_para_results_v3.npz'
        numpy.savez(out_file_name, acc=saved_acc, loss=saved_loss, trainloss=saved_training_loss,
                    testloss=saved_test_loss)
    plot_results(acc, loss, args4.num_epoch, legends, scheme)
