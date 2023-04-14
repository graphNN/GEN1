from SALib.sample import saltelli
from SALib.analyze import sobol, fast, morris
import matplotlib.pyplot as plot

import dgl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

import random
import time
import argparse

from GEN_models import GEN
from GEN_utils import edge_rand_prop, consis_loss, propagate_adj, edge_sample, random_splits
from GEN_utils import load_data_citation, pro_features_augmentation
from dgl.data import CoauthorPhysicsDataset, CoauthorCSDataset, CoraFullDataset


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', help='cora citeseer')

parser.add_argument('--num_layers', type=int, default=2, help='')
parser.add_argument('--edge_sample', type=float, default=1, help='edge sampling rate.')
parser.add_argument('--feat_sample', type=float, default=0.7, help='edge sampling rate.')
parser.add_argument('--num_graphs', type=int, default=4, help='')
parser.add_argument('--input_dropout', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_dropout', type=float, default=0.6,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--tem', type=float, default=0.3, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1.5, help='Lamda')
parser.add_argument('--a', type=float, default=0.1, help='LGEN alpha')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--feat_aug', type=bool, default=True)
parser.add_argument('--only_aug', type=bool, default=False)

parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')

args = parser.parse_args()

device = torch.device("cuda:0")


def evaluate(X):

    A, features, labels, train_mask, val_mask, test_mask, adj = load_data_citation(args.dataset)
    g = dgl.from_scipy(adj).to(device)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    mask = [train_mask, val_mask, test_mask]

    if args.feat_aug:
        torch.manual_seed(99)
        cont = pro_features_augmentation(features, labels, mask, X[0])
        if args.only_aug:
            features = cont
        else:
            features = torch.cat([features, cont], dim=1)

    adj_list = []
    feat_list = []

    for i in range(int(X[2]) - 1):  # num_graphs
        new_g = edge_rand_prop(dgl.remove_self_loop(g), 1.-X[1])
        new_g = dgl.add_self_loop(new_g)
        new_adj = propagate_adj(new_g.adj(scipy_fmt='csr'))
        adj_list.append(new_adj.to(device))

    new_g = edge_sample(dgl.remove_self_loop(g), 1.-X[1], train_mask, labels)
    new_g = dgl.add_self_loop(new_g)
    new_adj = propagate_adj(new_g.adj(scipy_fmt='csr'))
    adj_list.append(new_adj.to(device))
    feat_list.append(features)

    test_accs = []
    for i in range(args.seed, args.seed + 5):
        # seed
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)

        model = GEN(input_dim=features.shape[1], hidden=args.hidden, classes=(int(labels.max()) + 1),
                     num_graphs=int(X[2]),
                     dropout=[X[6], X[7]],
                     num_layers=args.num_layers, activation=True, alpha=X[3], use_bn=False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.to(device)

        t0 = time.time()
        best_acc, best_val_acc, best_test_acc, best_val_loss = 0, 0, 0, float("inf")
        for epoch in range(args.epochs):
            model.train()
            t1 = time.time()
            outputs = model(adj_list, feat_list)
            outputs_ = F.log_softmax(outputs, dim=1)
            train_loss = F.cross_entropy(outputs_[train_mask], labels[train_mask])

            cosi_loss = consis_loss(outputs_, X[4], X[5])

            optimizer.zero_grad()
            (train_loss + cosi_loss).backward()
            optimizer.step()

            model.eval()  # val
            with torch.no_grad():
                outputs = model(adj_list, feat_list)
                outputs_ = F.log_softmax(outputs, dim=1)

                train_loss_ = F.cross_entropy(outputs_[train_mask], labels[train_mask]).item()
                train_pred = outputs_[train_mask].max(dim=1)[1].type_as(labels[train_mask])
                train_correct = train_pred.eq(labels[train_mask]).double()
                train_correct = train_correct.sum()
                train_acc = (train_correct / len(labels[train_mask])) * 100

                val_loss = F.cross_entropy(outputs_[val_mask], labels[val_mask]).item()
                val_pred = outputs_[val_mask].max(dim=1)[1].type_as(labels[val_mask])
                correct = val_pred.eq(labels[val_mask]).double()
                correct = correct.sum()
                val_acc = (correct / len(labels[val_mask])) * 100

            model.eval()  # test
            with torch.no_grad():
                # outputs = model(g_list, features, order_attn)
                # outputs = F.log_softmax(outputs, dim=1)
                # test_loss = F.cross_entropy(outputs_[test_mask], labels[test_mask]).item()
                test_pred = outputs_[test_mask].max(dim=1)[1].type_as(labels[test_mask])
                correct = test_pred.eq(labels[test_mask]).double()
                correct = correct.sum()
                test_acc = (correct / len(labels[test_mask])) * 100

                # print(outputs_.sum())
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_test_acc = test_acc
                    bad_epoch = 0

                else:
                    bad_epoch += 1

            # epoch_time = time.time() - t1
            # if (epoch + 1) % 50 == 0:
            #     print('Epoch: {:3d}'.format(epoch), 'Train loss: {:.4f}'.format(train_loss_),
            #           '|Train accuracy: {:.2f}%'.format(train_acc), '||Val loss: {:.4f}'.format(val_loss),
            #           '||Val accuracy: {:.2f}%'.format(val_acc), '||Time: {:.2f}'.format(epoch_time))

            if bad_epoch == args.patience:
                break

        # _time = time.time() - t0
        # print('\n', 'Test accuracy:', best_test_acc)
        # print('Time of training model:', _time)
        # print('End of the training !')
        # print('-' * 100)

        test_accs.append(best_test_acc.item())

    # print(test_accs)
    # print(f'Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}')

    return np.mean(test_accs)


problem = {
    'num_vars': 8,
    'names': ['p1', 'p2', 'C', 'a', 'T', 'lam', 'drop1', 'drop2'],
    'bounds': [[0, 1], [0, 1], [1, 5], [0.1, 0.5], [0.1, 0.5], [0.5, 1.5], [0.3, 0.8], [0.3, 0.8]],
}

# param_values = saltelli.sample(problem, 128)  # 256  128
# # print(param_values.shape)
# Y = np.zeros([param_values.shape[0]])
# for i, X in enumerate(param_values):
#     Y[i] = evaluate(X)
#
# torch.save(param_values, 'param_values1.pkl')
# torch.save(Y, 'saY0.pkl')
# Si = sobol.analyze(problem, Y, print_to_console=True)


def plot_(S):
    # print(Sis)
    x = ['$p_{feat}$', '$p_{stru}$', 'C', f'{chr(945)}', 'T', '$\lambda$', 'drop1', 'drop2']
    error_kw = {'elinewidth': 4, 'ecolor': 'orange', 'capsize': 6}
    plot.figure(figsize=(8, 6))
    plot.grid(linestyle="-.")
    plot.bar(x, S['S1'], width=0.7, yerr=S['S1_conf'], error_kw=error_kw)  # S1 // ST
    plot.xlabel('Hyperparameters')
    plot.ylabel('First-order sensitivity indices')
    plot.savefig('sa_s1.pdf')


Y = torch.load('saY0.pkl')  # Y
Si = sobol.analyze(problem, Y, print_to_console=True)
print(Si)
plot_(Si)
plot.show()

