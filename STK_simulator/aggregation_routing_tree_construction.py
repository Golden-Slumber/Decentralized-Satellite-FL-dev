import copy

import numpy
from STK_simulator.constellation_config import WalkerStarConnectivity


def floyd_shortest_path(connectivity_matrix):
    n = len(connectivity_matrix)
    path_matrix = copy.deepcopy(connectivity_matrix)
    routing_matrix = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(n):
            if path_matrix[i][j] > 0:
                routing_matrix[i][j] = j + 1

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if path_matrix[i][k] > 0 and path_matrix[k][j] > 0 and (
                        path_matrix[i][k] + path_matrix[k][j] < path_matrix[i][j] or path_matrix[i][j] == -1):
                    path_matrix[i][j] = path_matrix[i][k] + path_matrix[k][j]
                    routing_matrix[i][j] = k + 1
    # print(path_matrix)
    # print(routing_matrix)
    return path_matrix, routing_matrix


def matrix_preprocessing(connectivity_matrix):
    n = len(connectivity_matrix)
    for i in range(n):
        for j in range(n):
            if connectivity_matrix[i][j] > 0:
                connectivity_matrix[i][j] = 1 / connectivity_matrix[i][j]
    total_weight = 0
    for i in range(n):
        for j in range(n):
            if connectivity_matrix[i][j] > 0:
                total_weight = total_weight + connectivity_matrix[i][j]
    for i in range(n):
        for j in range(n):
            if connectivity_matrix[i][j] > 0:
                connectivity_matrix[i][j] = connectivity_matrix[i][j] + total_weight * n
    return connectivity_matrix


def edge_set_construction(connectivity_matrix):
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if connectivity_matrix[i][j] > 0:
                edge_list.append((i, j, connectivity_matrix[i][j]))
    return edge_list


def MDST_construction(connectivity_matrix):
    n = len(connectivity_matrix)
    connectivity_matrix = matrix_preprocessing(connectivity_matrix)
    print(connectivity_matrix)
    distance_matrix, _ = floyd_shortest_path(connectivity_matrix)
    rk_matrix = [[0 for i in range(n)] for j in range(n)]
    edge_list = edge_set_construction(connectivity_matrix)
    print(edge_list)

    for i in range(n):
        for j in range(n):
            rk_matrix[i][j] = j
        for j in range(n):
            min_dis_idx = rk_matrix[i][j]
            for k in range(j + 1, n):
                if distance_matrix[i][rk_matrix[i][k]] < distance_matrix[i][min_dis_idx]:
                    rk_matrix[i][j] = k
                    rk_matrix[i][k] = min_dis_idx
                    min_dis_idx = k

    # in case the center is on some point
    center_p = 0
    diameter_p = 1e10
    for i in range(n):
        if distance_matrix[i][rk_matrix[i][n - 1]] * 2 < diameter_p:
            diameter_p = distance_matrix[i][rk_matrix[i][n - 1]] * 2
            center_p = i

    # in case the center is on some edge
    center_l, center_r, dis_l, dis_r, diameter_e = 0, 0, 0, 0, 1e10
    for edge in edge_list:
        u, v, w = edge[0], edge[1], edge[2]
        p = n - 1
        for i in range(2, n + 1):
            if distance_matrix[v][rk_matrix[u][n - i]] > distance_matrix[v][rk_matrix[u][p]]:
                if distance_matrix[u][rk_matrix[u][n - i]] + distance_matrix[v][rk_matrix[u][p]] + w < diameter_e:
                    diameter_e = distance_matrix[u][rk_matrix[u][n - i]] + distance_matrix[v][rk_matrix[u][p]] + w
                    center_l, center_r = u, v
                    dis_l = (distance_matrix[v][rk_matrix[u][p]] - distance_matrix[u][rk_matrix[u][n - i]] + w) / 2
                    dis_r = w - dis_l
                p = n - i

    shortest_path_tree = [[0 for i in range(n)] for j in range(n)]
    average_matrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                shortest_path_tree[i][j] = 1e10
            else:
                average_matrix[i][j] = 1

    if diameter_p <= diameter_e:
        for i in range(n):
            for edge in edge_list:
                u, v, w = edge[0], edge[1], edge[2]
                if shortest_path_tree[center_p][u] + w == distance_matrix[center_p][v] and shortest_path_tree[center_p][
                    u] + w < shortest_path_tree[center_p][v]:
                    shortest_path_tree[center_p][v] = shortest_path_tree[center_p][u] + w
                    average_matrix[u][v] = 1
                if shortest_path_tree[center_p][v] + w == distance_matrix[center_p][u] and shortest_path_tree[center_p][
                    v] + w < shortest_path_tree[center_p][u]:
                    shortest_path_tree[center_p][u] = shortest_path_tree[center_p][v] + w
                    average_matrix[v][u] = 1
    else:
        print('flag')
        augmented_connectivity_matrix = [[0 for i in range(n + 1)] for j in range(n + 1)]
        for i in range(n):
            for j in range(n):
                augmented_connectivity_matrix[i][j] = connectivity_matrix[i][j]
        for i in range(n + 1):
            if i == n:
                augmented_connectivity_matrix[i][i] = 0
            elif i == center_l:
                augmented_connectivity_matrix[i][n] = dis_l
                augmented_connectivity_matrix[n][i] = dis_l
            elif i == center_r:
                augmented_connectivity_matrix[i][n] = dis_r
                augmented_connectivity_matrix[i][n] = dis_r
            else:
                augmented_connectivity_matrix[i][n] = -1
                augmented_connectivity_matrix[n][i] = -1
        edge_list.append((n, center_l, dis_l))
        edge_list.append((n, center_r, dis_r))
        augmented_distance_matrix, _ = floyd_shortest_path(augmented_connectivity_matrix)
        center_p = n

        augmented_shortest_path_tree = [[0 for i in range(n + 1)] for j in range(n + 1)]
        for i in range(n+1):
            for j in range(n+1):
                if i != j:
                    augmented_shortest_path_tree[i][j] = 1e10
        for i in range(n+1):
            for edge in edge_list:
                u, v, w = edge[0], edge[1], edge[2]
                if augmented_shortest_path_tree[center_p][u] + w == augmented_distance_matrix[center_p][v] and augmented_shortest_path_tree[center_p][
                    u] + w < augmented_shortest_path_tree[center_p][v]:
                    augmented_shortest_path_tree[center_p][v] = augmented_shortest_path_tree[center_p][u] + w
                    if u != n and v != n:
                        average_matrix[u][v] = 1
                if augmented_shortest_path_tree[center_p][v] + w == augmented_distance_matrix[center_p][u] and augmented_shortest_path_tree[center_p][
                    v] + w < augmented_shortest_path_tree[center_p][u]:
                    augmented_shortest_path_tree[center_p][u] = augmented_shortest_path_tree[center_p][v] + w
                    if u != n and v != n:
                        average_matrix[v][u] = 1
    return average_matrix

def simplified_MDST_construction(connectivity_matrix):
    n = len(connectivity_matrix)
    for i in range(n):
        for j in range(n):
            if connectivity_matrix[i][j] > 0:
                connectivity_matrix[i][j] = 1 / connectivity_matrix[i][j]

    total_weight = 0
    for i in range(n):
        for j in range(n):
            if connectivity_matrix[i][j] > 0:
                total_weight = total_weight + connectivity_matrix[i][j]
    for i in range(n):
        for j in range(n):
            if connectivity_matrix[i][j] > 0:
                connectivity_matrix[i][j] = connectivity_matrix[i][j] + total_weight * n
    path_matrix, routing_matrix = floyd_shortest_path(connectivity_matrix)
    print(path_matrix)
    print(routing_matrix)
    radius_list = []
    # for i in range(n):
    #     radius_list.append(max(path_matrix[i]))
    # center_plane = numpy.argmin(radius_list)
    # print(center_plane + 1)
    return path_matrix, routing_matrix


if __name__ == '__main__':
    connectivity_matrix = WalkerStarConnectivity
    n = len(connectivity_matrix)
    for i in range(n):
        for j in range(n):
            if i != j and connectivity_matrix[i][j] == 0.0:
                connectivity_matrix[i][j] = -1
    # floyd_shortest_path(connectivity_matrix)
    # simplified_MDST_construction(connectivity_matrix)
    print(connectivity_matrix)
    average_matrix = MDST_construction(connectivity_matrix)
    print(average_matrix)