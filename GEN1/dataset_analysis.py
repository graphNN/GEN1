import torch
import os
from GEN_utils import load_data_citation
from load_geom import load_geom
import dgl

import numpy as np
import matplotlib.pyplot as plt


def data_analysis(dataset_name):
    dataset = dataset_name  # chameleon, film, squirrel, cora, citeseer, coauthorCS
    if dataset in {'chameleon', 'film', 'squirrel'}:
        path = os.getcwd() + '/splits/'
        dataset_split = path + f'{dataset}_split_0.6_0.2_{0}.npz'
        g, features, labels, train_mask, val_mask, test_mask = load_geom(dataset, dataset_split, train_percentage=None,
                                                                         val_percentage=None,
                                                                         embedding_mode=None, embedding_method=None,
                                                                         embedding_method_graph=None,
                                                                         embedding_method_space=None)
    if dataset in {'cora', 'citeseer'}:
        A, features, labels, train_mask, val_mask, test_mask, adj = load_data_citation(dataset)
        g = dgl.from_scipy(adj)
        g = dgl.remove_self_loop(g)

    src, dst = g.edges()

    print('classes and features:', int(labels.max()) + 1, features.shape[1])
    # print('train, val and test:', len(train_mask), len(val_mask), len(test_mask))
    print('nodes:', len(g.nodes()))
    print('edges:', len(src))
    h = 0
    for i in range(len(src)):
        if labels[src[i]] == labels[dst[i]]:
            h = h + 1
    print('h:', h / len(src))

data = 'cora'
data_analysis(data)


def time_bar(name, datasets, y):
    fig, ax = plt.subplots()
    n = len(datasets)
    x = np.arange(n) + 1
    plt.bar(x - 0.4, y[0], alpha=0.6, width=0.2, label=name[0])
    plt.bar(x - 0.2, y[1], alpha=0.6, width=0.2, label=name[1])
    plt.bar(x, y[2], alpha=0.6, width=0.2, label=name[2])
    plt.bar(x + 0.2, y[3], alpha=0.6, width=0.2, label=name[3])

    plt.legend(loc='upper left')
    plt.xlabel('Datasets')
    plt.ylabel('Time(s)')
    plt.xticks(x)
    ax.set_xticklabels(datasets)

    for aa, bb in zip(x - 0.4, y[0]):
        plt.text(aa, bb, '%.2f' % bb, ha='center', va='bottom', fontsize=8)
    for aa, bb in zip(x - 0.2, y[1]):
        plt.text(aa, bb, '%.2f' % bb, ha='center', va='bottom', fontsize=8)
    for aa, bb in zip(x, y[2]):
        plt.text(aa, bb, '%.2f' % bb, ha='center', va='bottom', fontsize=8)
    for aa, bb in zip(x + 0.2, y[3]):
        plt.text(aa, bb, '%.2f' % bb, ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    plt.savefig('time.pdf')
    plt.show()


# run time
# name = ['GCN', 'GAT', 'APPNP', 'GEN']
# datasets = ['Cora', 'Citeseer', 'Chameleon']
#
# y0 = [ , , ]
# y1 = [ , , ]
# y2 = [ , , ]
# y3 = [ , , ]
#
# y = []
# y.append(y0)
# y.append(y1)
# y.append(y2)
# y.append(y3)

# time_bar(name, datasets, y)


