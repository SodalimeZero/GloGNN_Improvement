import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from models.GloGNN import MLPNORM


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class MLPNORM_IMPROVE(MLPNORM):
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta,
                 norm_func_id, norm_layers, orders, orders_func_id, device, out_channels, num_layers, z1, z2, 
                 without_initial=False, without_topology=False):
        MLPNORM.__init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta,
                norm_func_id, norm_layers, orders, orders_func_id, device)
        self.W = nn.Linear(2*nhid, nhid)
        self.mlp_final = MLP(nhid, nhid, out_channels, num_layers, dropout=dropout)
        self.z1 = torch.tensor(z1).to(device)
        self.z2 = torch.tensor(z2).to(device)
        self.without_initial = without_initial
        self.without_topology = without_topology
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()
        self.orders_weight = Parameter(
            (torch.ones(self.orders, 1) / self.orders).to(self.device), requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.diag_weight = Parameter(
            (torch.ones(self.nclass, 1) / self.nclass).to(self.device), requires_grad=True
        )

    def forward(self, x, adj):
        xX = self.fc1(x)
        # x = self.bn1(x)
        xA = self.fc4(adj)
        if self.without_initial:
            x = F.relu(self.delta * xX + (1-self.delta) * xA)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.fc3(x))
            # x = self.bn2(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.fc2(x)
        else:
            x = F.relu(torch.cat((xA, xX), axis=-1))
            x = self.W(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(x + xA + xX)
            x = self.mlp_final(x, input_tensor=True)
        h0 = x
        for _ in range(self.norm_layers):
            x = self.norm(x, h0, adj)
        return x
    
    def norm_func1(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1.0 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x) # H^l的转置 * H^l
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res) # Q^(l+1)的后半段
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res) # Q^(l+1)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        first_order, sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.z1 * first_order + self.z2 * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    # glognn++
    def norm_func2(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # tmp_orders = torch.sparse.spmm(adj, res)
        if self.without_topology:
            tmp_orders = adj.matmul(res) # 矩阵乘法
            sum_orders = tmp_orders * self.orders_weight[0]
            for i in range(1, self.orders):
                # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
                tmp_orders = adj.matmul(tmp_orders)
                sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
            return 0, sum_orders
        else:
            # sum_orders = torch.zeros_like(res)  # 初始化总和，与节点特征形状相同
            # for k in range(1, self.orders + 1):
            tmp_orders = adj.matmul(res)  # 第一跳，计算邻接矩阵和节点特征的矩阵乘法
            sum_orders = tmp_orders * self.orders_weight[0]  # 第一跳的加权结果
            first_order = sum_orders
            if self.orders == 2:
                tmp_orders = adj.matmul(tmp_orders)
                sum_orders += tmp_orders * self.orders_weight[1]
                first_order = sum_orders
            elif self.orders > 2:
                for i in range(3, self.orders + 1):
                    tmp_orders = adj.matmul(tmp_orders)  # 第 i 跳，多次邻接矩阵和之前结果的矩阵乘法
                    sum_orders += tmp_orders * self.orders_weight[i - 1]  # 第 i 跳的加权结果

            return first_order, sum_orders

    def order_func3(self, x, res, adj):
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        orders_para = torch.transpose(orders_para, 0, 1)
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders