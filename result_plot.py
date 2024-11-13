import sys
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import PercentFormatter
import numpy

from constants import *

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
    out_file_name = home_dir + 'Outputs/EuroSat_SNN_InterPlane_Comparison_Repeat_' + str(0) + '_results.npz'
    acc = numpy.zeros((3, 1, 50))
    loss = numpy.zeros((3, 1, 50))

    npz_file = numpy.load(out_file_name, allow_pickle=True)
    saved_acc = npz_file['acc']
    saved_loss = npz_file['loss']

    for i in range(3):
        for j in range(50):
            if i == 1:
                acc[i, 0, j] = saved_acc[i-1, j]
                loss[i, 0, j] = saved_loss[i-1, j]
            else:
                acc[i, 0, j] = saved_acc[i, j]
                loss[i, 0, j] = saved_loss[i, j]
    legends = ['RelaySum (Proposed Inter-Plane Aggregation)', 'Gossip (Naive Inter-Plane Aggregation)',
               'All-Reduce (Baseline)']
    scheme = 'test'

    plot_results(acc, loss, 50, legends, scheme)
