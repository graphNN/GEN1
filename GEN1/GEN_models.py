
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


# graph Convolution
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(GraphConvolution, self).__init__()
        self.out_dim = out_dim

        self.weight = nn.parameter.Parameter(torch.FloatTensor(input_dim, out_dim))
        self.init_parameter()

    def init_parameter(self):
        stdv = 1. / math.sqrt(self.out_dim)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, x, alpha):
        h = (1-alpha) * torch.spmm(adj, x) + alpha * x
        h = torch.mm(h, self.weight)

        return h


class GEN(nn.Module):
    def __init__(self, input_dim, hidden, classes, dropout, num_graphs, num_layers, alpha, activation, use_bn):
        super(GEN, self).__init__()
        self.dropout = dropout
        self.num_graphs = num_graphs
        self.num_layers = num_layers
        self.alpha = alpha
        self.activation = activation
        self.use_bn = use_bn

        if use_bn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.layer = nn.ModuleList()
            for j in range(num_graphs):
                if i == 0:
                    if j == 0:
                        self.layer.append(GraphConvolution(input_dim, hidden))
                    else:
                        self.layer.append(GraphConvolution(input_dim, hidden))
                elif 0 < i < num_layers - 1:
                    self.layer.append(GraphConvolution(hidden, hidden))
                else:
                    if j == 0:
                        self.layer.append(GraphConvolution(hidden, classes))
                    else:
                        self.layer.append(GraphConvolution(hidden, classes))

            self.convs.append(self.layer)
        # self.conv_params = list(self.convs.parameters())

    def forward(self, adj_list, x_list):
        if self.use_bn:
            x_list[0] = self.bn1(x_list[0])
        h = F.dropout(x_list[0], self.dropout[0], self.training)

        for i in range(self.num_layers):
            h_layer = []
            for j in range(self.num_graphs):
                h1 = self.convs[i][j](adj_list[j], h, self.alpha)
                h_layer.append(h1)
            h = sum(h_layer) / self.num_graphs
            if i < self.num_layers - 1:
                if self.use_bn:
                    h = self.bn2(h)
                if self.activation:
                    h = F.relu(h)
                h = F.dropout(h, self.dropout[1], training=self.training)

        return h


class shallow_GEN(nn.Module):
    def __init__(self, input_dim, hidden, classes, dropout, num_graphs, num_layers, alpha, activation, use_bn):
        super(shallow_GEN, self).__init__()
        self.dropout = dropout
        self.num_graphs = num_graphs
        self.num_layers = num_layers
        self.alpha = alpha
        self.activation = activation
        self.use_bn = use_bn

        if use_bn:
            self.bn1 = nn.BatchNorm1d(input_dim)
            self.bn2 = nn.BatchNorm1d(hidden)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.layer = nn.ModuleList()
            for j in range(num_graphs):
                if i == 0:
                    if j == 0:
                        self.layer.append(GraphConvolution(input_dim, hidden))
                    else:
                        self.layer.append(GraphConvolution(input_dim, hidden))
                elif 0 < i < num_layers - 1:
                    self.layer.append(GraphConvolution(hidden, hidden))
                else:
                    if j == 0:
                        self.layer.append(GraphConvolution(hidden, classes))
                    else:
                        self.layer.append(GraphConvolution(hidden, classes))

            self.convs.append(self.layer)
        # self.conv_params = list(self.convs.parameters())

    def forward(self, adj_list, x_list):
        if self.use_bn:
            x_list[0] = self.bn1(x_list[0])
        h_list = []
        for i in range(len(x_list)):
            h_list.append(F.dropout(x_list[i], self.dropout[0], self.training))

        for i in range(self.num_layers):
            for j in range(self.num_graphs):
                h1 = self.convs[i][j](adj_list[j], h_list[j], self.alpha)
                h_list[j] = h1
            if i < self.num_layers - 1:
                if self.use_bn:
                    for j in range(self.num_graphs):
                        h_list[j] = self.bn2(h_list[j])
                if self.activation:
                    for j in range(self.num_graphs):
                        h_list[j] = F.relu(h_list[j])
                for j in range(self.num_graphs):
                    h_list[j] = F.dropout(h_list[j], self.dropout[1], training=self.training)
        h = sum(h_list) / self.num_graphs

        return h

