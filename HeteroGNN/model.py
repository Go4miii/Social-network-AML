#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/6/11 15:41 
# @Author : Iker Zhe 
# @Version：V 0.1
# @File : model.py
# @desc :

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, HeteroGraphConv, GATConv, SAGEConv


class HeteroGNN(nn.Module):
    def __init__(self, in_tx_feats, in_wallet_feats, hidden_size, out_tx_feats, out_wallet_feats):
        super(HeteroGNN, self).__init__()
        self.tx_linear = nn.Linear(in_tx_feats, hidden_size)
        self.wallet_linear = nn.Linear(in_wallet_feats, hidden_size)

        self.conv1 = HeteroGraphConv({
            'relates': GraphConv(hidden_size, hidden_size),
            'linked_to': GraphConv(hidden_size, hidden_size),
            'linked_to_rev': GraphConv(hidden_size, hidden_size),
            'relates_rev': GraphConv(hidden_size, hidden_size)
        }, aggregate='sum')

        self.conv2 = HeteroGraphConv({
            'relates': GraphConv(hidden_size, out_tx_feats),
            'linked_to': GraphConv(hidden_size, out_wallet_feats),
            'linked_to_rev': GraphConv(hidden_size, out_tx_feats),
            'relates_rev': GraphConv(hidden_size, out_wallet_feats)
        }, aggregate='sum')

    def forward(self, g, inputs):
        h_dict = {}
        h_dict['transaction'] = F.relu(self.tx_linear(inputs['transaction']))
        h_dict['wallet'] = F.relu(self.wallet_linear(inputs['wallet']))

        h_dict = self.conv1(g, h_dict)
        h_dict = {k: F.relu(h) for k, h in h_dict.items()}
        h_dict = self.conv2(g, h_dict)
        return h_dict

class HeteroGNN(nn.Module):
    def __init__(self, in_tx_feats, in_wallet_feats, hidden_size, out_tx_feats, out_wallet_feats, num_heads=4):
        super(HeteroGNN, self).__init__()
        self.tx_linear = nn.Linear(in_tx_feats, hidden_size)
        self.wallet_linear = nn.Linear(in_wallet_feats, hidden_size)

        # 使用 GAT 和 GraphSAGE 作为卷积层
        self.gat_conv1 = HeteroGraphConv({
            'relates': GATConv(hidden_size, hidden_size, num_heads, allow_zero_in_degree=True),
            'linked_to': GATConv(hidden_size, hidden_size, num_heads, allow_zero_in_degree=True),
            'linked_to_rev': GATConv(hidden_size, hidden_size, num_heads, allow_zero_in_degree=True),
            'relates_rev': GATConv(hidden_size, hidden_size, num_heads, allow_zero_in_degree=True)
        }, aggregate='mean')

        self.sage_conv1 = HeteroGraphConv({
            'relates': SAGEConv(hidden_size * num_heads, hidden_size, 'mean'),
            'linked_to': SAGEConv(hidden_size * num_heads, hidden_size, 'mean'),
            'linked_to_rev': SAGEConv(hidden_size * num_heads, hidden_size, 'mean'),
            'relates_rev': SAGEConv(hidden_size * num_heads, hidden_size, 'mean')
        }, aggregate='sum')

        self.gat_conv2 = HeteroGraphConv({
            'relates': GATConv(hidden_size, hidden_size, num_heads, allow_zero_in_degree=True),
            'linked_to': GATConv(hidden_size, hidden_size, num_heads, allow_zero_in_degree=True),
            'linked_to_rev': GATConv(hidden_size, hidden_size, num_heads, allow_zero_in_degree=True),
            'relates_rev': GATConv(hidden_size, hidden_size, num_heads, allow_zero_in_degree=True)
        }, aggregate='mean')

        self.sage_conv2 = HeteroGraphConv({
            'relates': SAGEConv(hidden_size * num_heads, out_tx_feats, 'mean'),
            'linked_to': SAGEConv(hidden_size * num_heads, out_wallet_feats, 'mean'),
            'linked_to_rev': SAGEConv(hidden_size * num_heads, out_tx_feats, 'mean'),
            'relates_rev': SAGEConv(hidden_size * num_heads, out_wallet_feats, 'mean')
        }, aggregate='sum')

    def forward(self, g, inputs):
        h_dict = {}
        h_dict['transaction'] = F.relu(self.tx_linear(inputs['transaction']))
        h_dict['wallet'] = F.relu(self.wallet_linear(inputs['wallet']))

        h_dict = self.gat_conv1(g, h_dict)
        h_dict = {k: F.relu(h.view(h.size(0), -1)) for k, h in h_dict.items()}

        h_dict = self.sage_conv1(g, h_dict)
        h_dict = {k: F.relu(h) for k, h in h_dict.items()}

        h_dict = self.gat_conv2(g, h_dict)
        h_dict = {k: F.relu(h.view(h.size(0), -1)) for k, h in h_dict.items()}

        h_dict = self.sage_conv2(g, h_dict)
        return h_dict



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        if self.weight is None:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
