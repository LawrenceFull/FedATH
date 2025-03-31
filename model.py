import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):

    def __init__(self, feat_dim, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feat_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, data, edge_scores=None):
        x, edge_index = data.x.float(), data.edge_index
        if edge_scores is None:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            embedding = F.dropout(x, p=self.dropout)
            x = self.conv2(embedding, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight=edge_scores)
            x = F.relu(x)
            embedding = F.dropout(x, p=self.dropout)
            x = self.conv2(embedding, edge_index, edge_weight=edge_scores)

        return x


    def rep_forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv2(x, edge_index)
        return x


class edge_mask(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout_rate):
        super(edge_mask, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        self.embedding_layer = nn.Linear(input_dim, hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate) 
        
        self.mask_net = mask_net(out_dim*2, 1)

        self.gcn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(1)])
        self.gcn_layers.append(GCNConv(hidden_dim, out_dim))

        self.sigmoid = nn.Sigmoid()


    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        
        h = self.embedding_layer(x)
        h = self.embedding_dropout(h)
        
        h = self.gcn_layers[0](h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate)
        h = self.gcn_layers[1](h, edge_index)

        link_score = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)
        link_score = self.mask_net(link_score)
        link_score = self.sigmoid(link_score)

        return link_score


class mask_net(nn.Module):
    def __init__(self, hidden_dim, out_dim, L=2):
        super(mask_net, self).__init__()
        mlp_list = [nn.Linear(hidden_dim//2**l, hidden_dim//2**(l+1), bias=True) for l in range(L)]
        batch_norms = [nn.BatchNorm1d(hidden_dim//2**(l+1)) for l in range(L)]
        mlp_list.append(nn.Linear(hidden_dim//2**L, out_dim, bias=True))

        self.mlp_list = nn.ModuleList(mlp_list)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.L = L
    
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.mlp_list[l](y)
            y = self.batch_norms[l](y)
            y = F.relu(y)
        y = self.mlp_list[self.L](y)
        return y

