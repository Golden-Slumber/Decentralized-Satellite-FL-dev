import numpy
from constants import WalkerStarConnectivity


def floyd_shortest_path(connectivity_matrix):
    n = len(connectivity_matrix)
    path_matrix = connectivity_matrix
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
    for i in range(n):
        radius_list.append(max(path_matrix[i]))
    center_plane = numpy.argmin(radius_list)
    print(center_plane + 1)


if __name__ == '__main__':
    connectivity_matrix = WalkerStarConnectivity
    n = len(connectivity_matrix)
    for i in range(n):
        for j in range(n):
            if i != j and connectivity_matrix[i][j] == 0.0:
                connectivity_matrix[i][j] = -1
    # floyd_shortest_path(connectivity_matrix)
    simplified_MDST_construction(connectivity_matrix)
