import sys
import numpy
from satellite_system import *
from learning_task import *
from utils import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

home_dir = './'
sys.path.append(home_dir)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def plot_results(acc, loss, iterations, legends, scheme, alpha):
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

    image_name = home_dir + 'Outputs/EuroSat_Comparison_demo_' + scheme + '_alpha_' + str(alpha) + '_accuracy.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EuroSat')
    parser.add_argument('--iterations', type=int, default=100, metavar='N')
    parser.add_argument('--intra-plane-iters', type=int, default=2, metavar='N')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--local-iters', type=int, default=3, metavar='N')
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)

    # centralized training
    # init_model = EuroSatCNN().to(device).state_dict()
    # training_task = EuroSatTask(args, train_set, init_model)
    # test_task = EuroSatTask(args, test_set, init_model)
    # for epoch in range(30):
    #     training_task.local_training()
    #     loss = training_task.get_training_loss()
    #     test_task.model_update(training_task.get_model())
    #     acc, _ = test_task.inference()
    #     print('Epoch {} \t Loss: {:.6f} \t Accuracy: {:.3f}%'.format(epoch, loss, 100. * acc))

    print(len(train_set))
    num_planes = 9
    num_satellites = 5
    datasize = int(len(train_set) / num_planes / num_satellites)
    datasize_array_by_plane = [datasize * numpy.ones(num_satellites).astype(int) for i in range(num_planes)]
    satellites_by_plane = [num_satellites for i in range(num_planes)]
    init_model = EuroSatCNN().to(device).state_dict()

    constellation = Constellation(num_planes, satellites_by_plane, train_set, test_set, datasize_array_by_plane,
                                  init_model, args)
    print('constellation initialization')

    # # inter-plane topology
    # connectivity_matrix = numpy.zeros((num_planes, num_planes))
    # # chain
    # for i in range(num_planes):
    #     connectivity_matrix[i, i] = 1
    #     for j in range(num_planes):
    #         if i - j == 1 or j - i == 1:
    #             connectivity_matrix[i, j] = 1
    # # binary tree
    # for i in range(num_planes):
    #     connectivity_matrix[i, i] = 1
    #     if int(2 * i) < num_planes:
    #         connectivity_matrix[i, int(2 * i)] = 1
    #         connectivity_matrix[int(2 * i), i] = 1
    #     if int(2 * i + 1) < num_planes:
    #         connectivity_matrix[i, int(2 * i + 1)] = 1
    #         connectivity_matrix[int(2 * i + 1), i] = 1
    # print(connectivity_matrix)
    # constellation.set_connectivity_matrix(connectivity_matrix)

    repeat = 1
    legends = ['RelaySum (Proposed Inter-Plane Aggregation)', 'Gossip (Naive Inter-Plane Aggregation)',
               'All-Reduce (Baseline)']

    acc = numpy.zeros((3, repeat, args.iterations))
    loss = numpy.zeros((3, repeat, args.iterations))
    saved_acc = numpy.zeros((3, args.iterations))
    saved_loss = numpy.zeros((3, args.iterations))

    print('Binary Tree Topology')
    for r in range(repeat):
        print('RelaySum Training')
        for p in constellation.plane_list:
            p.relay_sum_initialization(num_planes)
        constellation.reset_constellation(init_model)
        for t in range(args.iterations):
            connectivity_matrix = fixed_binary_tree_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(RELAYSUM)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[0, r, t] = constellation.test_accuracy[t]
            loss[0, r, t] = constellation.convergence_error[t]
            saved_acc[0, t] = constellation.test_accuracy[t]
            saved_loss[0, t] = constellation.convergence_error[t]

        constellation.reset_constellation(init_model)
        print('Gossip Training')
        for t in range(args.iterations):
            connectivity_matrix = fixed_binary_tree_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(GOSSIP)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[1, r, t] = constellation.test_accuracy[t]
            loss[1, r, t] = constellation.convergence_error[t]
            saved_acc[1, t] = constellation.test_accuracy[t]
            saved_loss[1, t] = constellation.convergence_error[t]

        constellation.reset_constellation(init_model)
        print('AllReduce Training')
        for t in range(args.iterations):
            connectivity_matrix = fixed_binary_tree_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(ALLREDUCE)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[2, r, t] = constellation.test_accuracy[t]
            loss[2, r, t] = constellation.convergence_error[t]
            saved_acc[2, t] = constellation.test_accuracy[t]
            saved_loss[2, t] = constellation.convergence_error[t]
        out_file_name = home_dir + 'Outputs/EuroSat_Binary_Tree_Comparison_Alpha_' + str(args.alpha) + '_Repeat_' + str(
            r) + '_results.npz'
        numpy.savez(out_file_name, acc=saved_acc, loss=saved_loss)
    plot_results(acc, loss, args.iterations, legends, 'Binary Tree', args.alpha)

    print('Chain Topology')
    for r in range(repeat):
        print('RelaySum Training')
        for p in constellation.plane_list:
            p.relay_sum_initialization(num_planes)
        constellation.reset_constellation(init_model)
        for t in range(args.iterations):
            connectivity_matrix = fixed_chain_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(RELAYSUM)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[0, r, t] = constellation.test_accuracy[t]
            loss[0, r, t] = constellation.convergence_error[t]
            saved_acc[0, t] = constellation.test_accuracy[t]
            saved_loss[0, t] = constellation.convergence_error[t]

        constellation.reset_constellation(init_model)
        print('Gossip Training')
        for t in range(args.iterations):
            connectivity_matrix = fixed_chain_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(GOSSIP)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[1, r, t] = constellation.test_accuracy[t]
            loss[1, r, t] = constellation.convergence_error[t]
            saved_acc[1, t] = constellation.test_accuracy[t]
            saved_loss[1, t] = constellation.convergence_error[t]

        constellation.reset_constellation(init_model)
        print('AllReduce Training')
        for t in range(args.iterations):
            connectivity_matrix = fixed_chain_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(ALLREDUCE)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[2, r, t] = constellation.test_accuracy[t]
            loss[2, r, t] = constellation.convergence_error[t]
            saved_acc[2, t] = constellation.test_accuracy[t]
            saved_loss[2, t] = constellation.convergence_error[t]
        out_file_name = home_dir + 'Outputs/EuroSat_Chain_Comparison_Alpha_' + str(args.alpha) + '_Repeat_' + str(
            r) + '_results.npz'
        numpy.savez(out_file_name, acc=saved_acc, loss=saved_loss)
    plot_results(acc, loss, args.iterations, legends, 'Chain', args.alpha)

    print('Ring Topology')
    for r in range(repeat):
        print('RelaySum Training')
        for p in constellation.plane_list:
            p.relay_sum_initialization(num_planes)
        constellation.reset_constellation(init_model)
        for t in range(args.iterations):
            connectivity_matrix = fixed_ring_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(RELAYSUM)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[0, r, t] = constellation.test_accuracy[t]
            loss[0, r, t] = constellation.convergence_error[t]
            saved_acc[0, t] = constellation.test_accuracy[t]
            saved_loss[0, t] = constellation.convergence_error[t]

        constellation.reset_constellation(init_model)
        print('Gossip Training')
        for t in range(args.iterations):
            connectivity_matrix = fixed_ring_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(GOSSIP)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[1, r, t] = constellation.test_accuracy[t]
            loss[1, r, t] = constellation.convergence_error[t]
            saved_acc[1, t] = constellation.test_accuracy[t]
            saved_loss[1, t] = constellation.convergence_error[t]

        constellation.reset_constellation(init_model)
        print('AllReduce Training')
        for t in range(args.iterations):
            connectivity_matrix = fixed_ring_topology(num_planes)
            constellation.set_connectivity_matrix(connectivity_matrix)
            constellation.constellation_training(ALLREDUCE)
            constellation.save_metric_v2(t)
        for t in range(args.iterations):
            acc[2, r, t] = constellation.test_accuracy[t]
            loss[2, r, t] = constellation.convergence_error[t]
            saved_acc[2, t] = constellation.test_accuracy[t]
            saved_loss[2, t] = constellation.convergence_error[t]
        out_file_name = home_dir + 'Outputs/EuroSat_Ring_Comparison_Alpha_' + str(args.alpha) + '_Repeat_' + str(
            r) + '_results.npz'
        numpy.savez(out_file_name, acc=saved_acc, loss=saved_loss)
    plot_results(acc, loss, args.iterations, legends, 'Ring', args.alpha)
