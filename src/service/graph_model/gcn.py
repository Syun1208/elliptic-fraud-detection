from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GINConv, GINEConv
from src.service.graph_decoder import *


class GCN(nn.Module):
    def __init__(
            self,
            num_features: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int= 1, 
            n_layers: int = 3, 
            dropout_rate: float = 0
            ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn_hidden = nn.ModuleList()
        self.n_layers = n_layers

        if n_layers == 1:
            self.gcn1 = GCNConv(num_features, embedding_dim)
        else:
            self.gcn1 = GCNConv(num_features, hidden_dim)
            for _ in range(n_layers-2): #first and last layer seperately
                self.gcn_hidden.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn2 = GCNConv(hidden_dim, embedding_dim)

        self.out = DecoderLinear(embedding_dim, output_dim)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.gcn_hidden:
                h = layer(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            h = self.gcn2(h, edge_index)
        out = self.out(h)

        return out, h