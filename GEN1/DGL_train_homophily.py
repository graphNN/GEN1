# graph neural networks on citation networks

import dgl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

import random
import time
import argparse

from DGL_models import dgl_gat, dgl_gcn, dgl_sage, dgl_appnp, dgl_gin, dgl_sgc
from GEN_utils import load_data_citation, random_splits
from dgl.data import CoauthorCSDataset


def select_model(model_name, features):
    if model_name == 'GAT':
        model = dgl_gat(input_dim=features.shape[1], out_dim=args.hidden, num_heads=[args.num_heads1, args.num_heads2],
                         num_classes=(int(labels.max())+1), dropout=[args.feat_dropout, args.attn_dropout])
    if model_name == 'GCN':
        model = dgl_gcn(input_dim=features.shape[1], nhidden=args.hidden, nclasses=(int(labels.max())+1))
    if model_name == 'SAGE':
        model = dgl_sage(input_dim=features.shape[1], nhidden=args.hidden, aggregator_type=args.sage_agg_type,
                         nclasses=(int(labels.max())+1))
    if model_name == 'APPNP':
        model = dgl_appnp(input_dim=features.shape[1], hidden=args.appnp_hidden, classes=(int(labels.max())+1),
                          k=args.K, alpha=args.alpha)
    if model_name == 'GIN':
        model = dgl_gin(input_dim=features.shape[1], hidden=args.gin_hidden, classes=(int(labels.max())+1),
                        aggregator_type=args.gin_agg_type)
    if model_name == 'SGC':
        model = dgl_sgc(input_dim=features.shape[1], hidden=args.hidden, classes=(int(labels.max()) + 1))

    return model


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', help='cora citeseer cs')
parser.add_argument('--model', type=str, default='GIN', help='GAT, GCN, GIN, APPNP, SAGE, SGC, TWIRLS')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--feat_dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn_dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--gin_hidden', type=int, default=8, help='MoNet:Number of hidden units.')
parser.add_argument('--appnp_hidden', type=int, default=8, help='APPNP:Number of hidden units.')
parser.add_argument('--num_heads1', type=int, default=8, help='GAT:number of head')
parser.add_argument('--num_heads2', type=int, default=1, help='GAT:number of head')

parser.add_argument('--gin_agg_type', type=str, default='mean', help='gin:sum mean max')
parser.add_argument('--sage_agg_type', type=str, default='gcn', help='sage:mean, gcn, pool, lstm')
parser.add_argument('--K', type=int, default=10, help='APPNP inter')  # paper:10
parser.add_argument('--alpha', type=float, default=0.1, help='APPNP alpha')

args = parser.parse_args()

device = torch.device("cuda:0")
dropout = [args.feat_dropout, args.attn_dropout]

if args.dataset in {'cora', 'citeseer'}:
    A, features, labels, train_mask, val_mask, test_mask, adj = load_data_citation(args.dataset)
    g = dgl.from_scipy(adj).to(device)
    # g = dgl.add_self_loop(g)

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
else:
    if args.dataset == 'cs':
        dataset = CoauthorCSDataset()
        g = dataset[0].to(device)
        g = dgl.add_self_loop(g)
        features = g.ndata['feat']
        labels = g.ndata['label']
        percls_trn = int(round(0.05 * len(labels) / int(labels.max() + 1)))
        val_lb = int(round(0.2 * len(labels)))
        train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb)
    else:
        print('dataset name error')
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

test_accs = []

for i in range(args.seed, args.seed + 10):
    # seed
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)

    model = select_model(args.model, features)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)

    t0 = time.time()
    best_acc, best_val_acc, best_test_acc, best_val_loss = 0, 0, 0, float("inf")
    for epoch in range(args.epochs):
        model.train()
        t1 = time.time()
        outputs = model(g, features)
        outputs_ = F.log_softmax(outputs, dim=1)
        train_loss = F.cross_entropy(outputs_[train_mask], labels[train_mask])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()  # val
        with torch.no_grad():
            outputs = model(g, features)
            outputs_ = F.log_softmax(outputs, dim=1)

            train_loss_ = F.cross_entropy(outputs_[train_mask], labels[train_mask]).item()
            train_pred = outputs_[train_mask].max(dim=1)[1].type_as(labels[train_mask])
            train_correct = train_pred.eq(labels[train_mask]).double()
            train_correct = train_correct.sum()
            train_acc = (train_correct / len(labels[train_mask])) * 100

            val_loss = F.cross_entropy(outputs[val_mask], labels[val_mask]).item()
            val_pred = outputs[val_mask].max(dim=1)[1].type_as(labels[val_mask])
            correct = val_pred.eq(labels[val_mask]).double()
            correct = correct.sum()
            val_acc = (correct / len(labels[val_mask])) * 100

        model.eval()  # test
        with torch.no_grad():
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
        if (epoch + 1) % 10 == 0:
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

print(test_accs)
print(f'Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}')

