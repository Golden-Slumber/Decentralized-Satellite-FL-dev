import numpy
import math


def Dirichlet_non_iid_distribution(targets, non_iid_alpha, n_devices, seed=0, n_auxi_devices=10):
    random_state = numpy.random.RandomState(seed=seed)
    num_indices = len(targets)
    num_classes = len(numpy.unique(targets))
    indices2targets = numpy.array(list(enumerate(targets)))
    random_state.shuffle(indices2targets)

    # partition indices
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_devices / n_auxi_devices)
    split_n_devices = [n_auxi_devices if idx < num_splits - 1 else n_devices - n_auxi_devices * (num_splits - 1) for idx
                       in range(num_splits)]
    split_ratios = [_n_devices / n_devices for _n_devices in split_n_devices]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_devices / n_devices * num_indices)
        splitted_targets.append(indices2targets[from_index:(num_indices if idx == num_splits - 1 else to_index)])
        from_index = to_index

    idx_batch = []
    for _targets in splitted_targets:
        _targets = numpy.array(_targets)
        _targets_size = len(_targets)

        _n_devices = min(n_auxi_devices, n_devices)
        n_devices = n_devices - n_auxi_devices

        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_devices):
            _idx_batch = [[] for _ in range(_n_devices)]
            for _class in range(num_classes):
                idx_class = numpy.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                try:
                    proportions = random_state.dirichlet(numpy.repeat(non_iid_alpha, _n_devices))
                    proportions = numpy.array(
                        [p * (len(idx_j) < _targets_size / _n_devices) for p, idx_j in zip(proportions, _idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (numpy.cumsum(proportions) * len(idx_class)).astype(int)[:-1]
                    _idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                                  zip(_idx_batch, numpy.split(idx_class, proportions))]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def time_varying_topology(num_planes, t):
    connectivity_matrix = numpy.zeros((num_planes, num_planes))
    if t % 3 == 0:
        for i in range(num_planes):
            connectivity_matrix[i, i] = 1
            if int(2 * i) < num_planes:
                connectivity_matrix[i, int(2 * i)] = 1
                connectivity_matrix[int(2 * i), i] = 1
            if int(2 * i + 1) < num_planes:
                connectivity_matrix[i, int(2 * i + 1)] = 1
                connectivity_matrix[int(2 * i + 1), i] = 1
    elif t % 3 == 1:
        for i in range(num_planes):
            connectivity_matrix[i, i] = 1
            for j in range(num_planes):
                if i - j == 1 or j - i == 1:
                    connectivity_matrix[i, j] = 1
    elif t % 3 == 2:
        for i in range(num_planes):
            connectivity_matrix[i, i] = 1
            for j in range(num_planes):
                if i - j == 1 or j - i == 1:
                    connectivity_matrix[i, j] = 1
        connectivity_matrix[0, num_planes - 1] = 1
        connectivity_matrix[num_planes - 1, 0] = 1
    return connectivity_matrix


def fixed_binary_tree_topology(num_planes):
    connectivity_matrix = numpy.zeros((num_planes, num_planes))
    for i in range(num_planes):
        connectivity_matrix[i, i] = 1
        if int(2 * i) < num_planes:
            connectivity_matrix[i, int(2 * i)] = 1
            connectivity_matrix[int(2 * i), i] = 1
        if int(2 * i + 1) < num_planes:
            connectivity_matrix[i, int(2 * i + 1)] = 1
            connectivity_matrix[int(2 * i + 1), i] = 1
    return connectivity_matrix

def fixed_chain_topology(num_planes):
    connectivity_matrix = numpy.zeros((num_planes, num_planes))
    for i in range(num_planes):
        connectivity_matrix[i, i] = 1
        for j in range(num_planes):
            if i - j == 1 or j - i == 1:
                connectivity_matrix[i, j] = 1
    return connectivity_matrix

def fixed_ring_topology(num_planes):
    connectivity_matrix = numpy.zeros((num_planes, num_planes))
    for i in range(num_planes):
        connectivity_matrix[i, i] = 1
        for j in range(num_planes):
            if i - j == 1 or j - i == 1:
                connectivity_matrix[i, j] = 1
    connectivity_matrix[0, num_planes - 1] = 1
    connectivity_matrix[num_planes - 1, 0] = 1
    return connectivity_matrix
