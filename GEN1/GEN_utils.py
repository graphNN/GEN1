import dgl
import torch
import scipy.sparse as sp
import numpy as np
import copy

import os
import sys
import pickle as pkl
import networkx as nx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def cate_grouping(labels):
    group = {}
    num_classes = labels.max().int() + 1
    for i in range(num_classes):
        group[i] = torch.nonzero(labels == i, as_tuple=True)[0]

    return group


def category_features(features, labels, mask):
    c = int(labels.max() + 1)
    group = cate_grouping(labels[mask[0]])

    cate_features_ = torch.zeros(c, features.shape[1]).to(features.device)

    for i in range(c):
        if len(group[i]) == 0:
            cate_features_[i] = features[mask[0]].mean(0)
        else:
            cate_features_[i] = features[mask[0]][group[i]].mean(0)

    return cate_features_


def pro_features_augmentation(features, labels, mask, p):
    cate_features_ = category_features(features, labels, mask)
    cont = torch.zeros(features.shape).to(features.device)
    p = torch.tensor(p)
    for i in range(features.shape[0]):
        if i in mask[2]:
            cont[i] = features[i] * 1
        elif i in mask[1]:
            if torch.bernoulli(p) == 1:
               cont[i] = cate_features_[labels[i]]
            else:
                cont[i] = features[i] * 1
        elif i in mask[0]:
            if torch.bernoulli(p) == 1:
               cont[i] = cate_features_[labels[i]]
            else:
               cont[i] = features[i] * 1
        else:
            cont[i] = features[i] * 1
    return cont


def edge_rand_prop(g, edge_drop_rate):  # 全部随机选
    torch.manual_seed(99)
    src, dst = g.edges()
    edges_num = len(src)

    drop_rates = torch.FloatTensor(np.ones(edges_num) * edge_drop_rate)
    masks = torch.as_tensor(torch.bernoulli(1. - drop_rates), dtype=torch.int).to(g.device)

    remove_edges_index = torch.nonzero(masks == 0, as_tuple=True)
    new_g = dgl.remove_edges(g, remove_edges_index[0])

    return new_g


def delete_from_tensor(ten, index):
    t = torch.arange(0, len(ten))
    t = t.tolist()
    for i in range(len(index)):
        t.remove(index[i])

    return torch.tensor(t)


def edge_sample(g, edge_sample_rate, mask, labels):  # 已知标签的保留类内边，未知的随机采样
    torch.manual_seed(99)
    device = g.device
    adj = g.adj(scipy_fmt='csr')
    temp = torch.zeros(adj.shape).to(device)
    dense_adj = torch.as_tensor(adj.todense()).to(device)
    classes = int(max(labels)) + 1
    cate = cate_grouping(labels[mask])

    for i in range(classes):
        l = copy.deepcopy(labels[mask])
        # l = delete_from_tensor(l, cate[i])
        index3 = cate[i].unsqueeze(0).T.repeat(1, len(l)).resize(len(cate[i]) * len(l))  # 某一类和所有类
        index4 = l.repeat(1, len(cate[i])).squeeze(0)
        index_ = [index3, index4]
        torch.index_put_(temp, index_, torch.ones(1).to(device) - 2)  # 标记训练集中所有的边

        index1 = cate[i].unsqueeze(0).T.repeat(1, len(cate[i])).resize(len(cate[i]) * len(cate[i]))  # 某一类和自己
        index2 = cate[i].repeat(1, len(cate[i])).squeeze(0)
        index = [index1, index2]
        torch.index_put_(temp, index, torch.ones(1).to(device))  # 标记类内边

    intra_adj = torch.where(temp > 0, dense_adj, 0)  # 训练集中的类内边
    index5 = torch.nonzero(intra_adj > 0, as_tuple=True)
    inter_adj = torch.where(temp < 0, dense_adj, 0)  # 训练集中的类间边
    index6 = torch.nonzero(inter_adj > 0, as_tuple=True)

    eids = g.edge_ids(index6[0].to(g.device), index6[1].to(g.device))
    g = dgl.remove_edges(g, eids)  # 移除训练集中的类间边
    new_g = edge_rand_prop(g, edge_sample_rate)    # 按概率采样边
    new_g = dgl.add_edges(new_g, index5[0].to(g.device), index5[1].to(g.device))  # 添加训练集中的类内边

    return new_g


def propagate_adj(adj):
    D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)
    A = sparse_mx_to_torch_sparse_tensor(A)

    return A


def consis_loss(logps, tem, lam):
    logps = torch.exp(logps)
    sharp_logps = (torch.pow(logps, 1. / tem) / torch.sum(torch.pow(logps, 1. / tem), dim=1, keepdim=True)).detach()
    loss = torch.mean((logps - sharp_logps).pow(2).sum(1)) * lam

    return loss


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_splits(labels, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,labels.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=len(labels))
    val_mask = index_to_mask(val_idx,size=len(labels))
    test_mask = index_to_mask(test_idx,size=len(labels))
    return train_mask, val_mask, test_mask


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    np.random.seed(42)
    torch.manual_seed(42)

    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


file_dir_citation = os.getcwd() + '/data'
def load_data_citation(dataset_str='cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(file_dir_citation, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(file_dir_citation, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    features = torch.FloatTensor(np.array(features.todense()))
    # features_norm = torch.FloatTensor(np.array(features_norm.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, features, labels, idx_train, idx_val, idx_test, adj

