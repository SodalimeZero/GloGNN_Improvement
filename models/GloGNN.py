import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init


class MLPNORM(nn.Module):
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta,
                 norm_func_id, norm_layers, orders, orders_func_id, device):
        super(MLPNORM, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nnodes, nhid)
        # self.bn1 = nn.BatchNorm1d(nhid)
        # self.bn2 = nn.BatchNorm1d(nhid)
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = torch.tensor(alpha).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.gamma = torch.tensor(gamma).to(device)
        self.delta = torch.tensor(delta).to(device)
        self.norm_layers = norm_layers
        self.orders = orders
        self.device = device
        self.class_eye = torch.eye(self.nclass).to(device)
        self.orders_weight = Parameter(
            (torch.ones(orders, 1) / orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix = Parameter(
            torch.DoubleTensor(nclass, orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix2 = Parameter(
            torch.DoubleTensor(orders, orders).to(device), requires_grad=True
        )
        self.diag_weight = Parameter(
            (torch.ones(nclass, 1) / nclass).to(device), requires_grad=True
        )
        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.orders_weight = Parameter(
            (torch.ones(self.orders, 1) / self.orders).to(self.device), requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.diag_weight = Parameter(
            (torch.ones(self.nclass, 1) / self.nclass).to(self.device), requires_grad=True
        )

    def forward(self, x, adj):
        # xd = F.dropout(x, self.dropout, training=self.training)
        # adjd = F.dropout(adj, self.dropout, training=self.training)
        xX = self.fc1(x)
        # x = self.bn1(x)
        xA = self.fc4(adj)
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        # x = self.bn2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
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
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

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
        tmp_orders = adj.matmul(res) # 矩阵乘法
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

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