import sys
import numpy
from satellite_system import *
from learning_task import *

home_dir = './'
sys.path.append(home_dir)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EuroSat')
    parser.add_argument('--iterations', type=int, default=50, metavar='N')
    parser.add_argument('--intra-plane-iters', type=int, default=1, metavar='N')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--local-iters', type=int, default=3, metavar='N')
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)

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

    # inter-plane topology
    connectivity_matrix = numpy.zeros((num_planes, num_planes))
    # chain
    # for i in range(num_planes):
    #     connectivity_matrix[i, i] = 1
    #     for j in range(num_planes):
    #         if i - j == 1 or j - i == 1:
    #             connectivity_matrix[i, j] = 1
    # binary tree
    for i in range(num_planes):
        connectivity_matrix[i, i] = 1
        if int(2 * i) < num_planes:
            connectivity_matrix[i, int(2 * i)] = 1
            connectivity_matrix[int(2 * i), i] = 1
        if int(2 * i + 1) < num_planes:
            connectivity_matrix[i, int(2 * i + 1)] = 1
            connectivity_matrix[int(2 * i + 1), i] = 1
    print(connectivity_matrix)
    constellation.set_connectivity_matrix(connectivity_matrix)

    print('Gossip Training')
    for t in range(args.iterations):
        constellation.constellation_training(GOSSIP)
        constellation.save_metric(t)

    constellation.reset_constellation(init_model)
    print('RelaySum Training')
    for p in constellation.plane_list:
        p.relay_sum_initialization(num_planes)
    constellation.reset_constellation(init_model)
    for t in range(args.iterations):
        constellation.constellation_training(RELAYSUM)
        constellation.save_metric(t)
