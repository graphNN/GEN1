
import os
import dgl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

import random
import time
import argparse
import optuna

from GEN_models import GEN, shallow_GEN
from GEN_utils import edge_rand_prop, consis_loss, propagate_adj
from load_geom import load_geom
from GEN_utils import edge_sample, pro_features_augmentation


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--dataset', type=str, default='squirrel', help='texas, cornell, wisconsin, chameleon')

parser.add_argument('--num_layers', type=int, default=3, help='')
parser.add_argument('--edge_sample', type=float, default=1, help='edge sampling rate.')
parser.add_argument('--feat_sample', type=float, default=0.7, help='feature sampling rate.')
parser.add_argument('--num_graphs', type=int, default=2, help='')
parser.add_argument('--input_dropout', type=float, default=0.3,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_dropout', type=float, default=0.1,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--tem', type=float, default=0.4, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=0.2, help='Lamda')
parser.add_argument('--alpha', type=float, default=0.2, help='LGEN alpha')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
parser.add_argument('--feat_aug', type=bool, default=True)
parser.add_argument('--only_aug', type=bool, default=False)
parser.add_argument('--if_deep', type=bool, default=True)

parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')


args = parser.parse_args()

device = torch.device("cuda:0")


def test_cat(trial):

    all_test_accs = []
    for j in range(10):
        path = os.getcwd() + '/splits/'
        dataset_split = path + f'{args.dataset}_split_0.6_0.2_{j}.npz'
        g, features, labels, train_mask, val_mask, test_mask = load_geom(args.dataset, dataset_split,
                                                                         train_percentage=None, val_percentage=None,
                                                                         embedding_mode=None, embedding_method=None,
                                                                         embedding_method_graph=None,
                                                                         embedding_method_space=None)

        g = dgl.add_self_loop(g).to(device)
        features = features.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        train_mask = train_mask.clone().detach().to(device)
        val_mask = val_mask.clone().detach().to(device)
        test_mask = test_mask.clone().detach().to(device)
        mask = [train_mask, val_mask, test_mask]

        if args.feat_aug:
            torch.manual_seed(99)
            cont = pro_features_augmentation(features, labels, mask, args.feat_sample)
            if args.only_aug:
                features = cont
            else:
                features = torch.cat([features, cont], dim=1)

        feats = []
        adj_list = []
        for i in range(args.num_graphs - 1):
            new_g = edge_rand_prop(dgl.remove_self_loop(g), 1-args.edge_sample)
            new_g = dgl.add_self_loop(new_g)
            new_adj = propagate_adj(new_g.adj(scipy_fmt='csr'))
            adj_list.append(new_adj.to(device))

        new_g = edge_sample(dgl.remove_self_loop(g), 1-args.edge_sample, train_mask, labels)
        new_g = dgl.add_self_loop(new_g)
        new_adj = propagate_adj(new_g.adj(scipy_fmt='csr'))
        adj_list.append(new_adj.to(device))
        if args.if_deep:
            feats.append(features)
        else:
            for i in range(args.num_graphs):
                feats.append(features)

        test_accs = []
        for i in range(args.seed, args.seed + 5):
            # seed
            random.seed(i)
            np.random.seed(i)
            torch.manual_seed(i)

            if args.if_deep:
                model = GEN(input_dim=features.shape[1], hidden=args.hidden, classes=(int(labels.max()) + 1),
                             num_graphs=args.num_graphs,
                             dropout=[args.input_dropout, args.hidden_dropout],
                             num_layers=args.num_layers, alpha=args.alpha, activation=True, use_bn=False)
            else:
                model = shallow_GEN(input_dim=features.shape[1], hidden=args.hidden, classes=(int(labels.max()) + 1),
                                     num_graphs=args.num_graphs,
                                     dropout=[args.input_dropout, args.hidden_dropout],
                                     num_layers=args.num_layers, alpha=args.alpha, activation=True, use_bn=False)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            model.to(device)

            t0 = time.time()
            best_acc, best_val_acc, best_test_acc, best_val_loss = 0, 0, 0, float("inf")
            for epoch in range(args.epochs):
                model.train()
                t1 = time.time()
                outputs = model(adj_list, feats)
                outputs_ = F.log_softmax(outputs, dim=1)
                train_loss = F.cross_entropy(outputs_[train_mask], labels[train_mask])
                cosi_loss = consis_loss(outputs_, args.tem, args.lam)

                optimizer.zero_grad()
                (train_loss + cosi_loss).backward()
                optimizer.step()

                model.eval()  # val
                with torch.no_grad():
                    outputs = model(adj_list, feats)
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
                    test_loss = F.cross_entropy(outputs_[test_mask], labels[test_mask]).item()
                    test_pred = outputs_[test_mask].max(dim=1)[1].type_as(labels[test_mask])
                    correct = test_pred.eq(labels[test_mask]).double()
                    correct = correct.sum()
                    test_acc = (correct / len(labels[test_mask])) * 100

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_test_acc = test_acc
                        bad_epoch = 0

                    else:
                        bad_epoch += 1

                epoch_time = time.time() - t1
                if (epoch + 1) % 50 == 0:
                    print('Epoch: {:3d}'.format(epoch), 'Train loss: {:.4f}'.format(train_loss_),
                          '|Train accuracy: {:.2f}%'.format(train_acc), '||Val loss: {:.4f}'.format(val_loss),
                          '||Val accuracy: {:.2f}%'.format(val_acc), '||Time: {:.2f}'.format(epoch_time))

                if bad_epoch == args.patience:
                    break

            _time = time.time() - t0
            print('\n', 'Test accuracy:', best_test_acc)
            print('Time of training model:', _time)
            print('End of the training !')
            print('-' * 100)

            test_accs.append(best_test_acc.item())
        all_test_accs.append(np.mean(test_accs))
        print(test_accs)
        print(f'Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}')
        print('-' * 100)

    print(all_test_accs)
    print(f'all Average test accuracy: {np.mean(all_test_accs)} ± {np.std(all_test_accs)}')
    return np.mean(all_test_accs)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(test_cat, n_trials=1)  # 搜索次数

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

