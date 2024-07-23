from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GINConv, GINEConv
from torch_geometric.nn.norm import LayerNorm
from src.service.graph_decoder import *


class GCN(nn.Module):
    def __init__(
            self,
            num_features: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            n_layers: int,
            dropout_rate: float = 0
            ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn_hidden = nn.ModuleList()
        self.gcn_norm = nn.ModuleList()
        self.n_layers = n_layers

        if n_layers == 1:
            self.ln1 = LayerNorm(num_features)
            self.gcn1 = GCNConv(num_features, embedding_dim)
        else:
            self.ln1 = LayerNorm(num_features)
            self.gcn1 = GCNConv(num_features, hidden_dim)
            for _ in range(n_layers - 2):
                self.gcn_hidden.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn2 = GCNConv(hidden_dim, embedding_dim)
        
        self.out = DecoderLinear(embedding_dim, output_dim, normalize=False)

    def forward(self, x, edge_index):
        h = self.ln1(x)
        h = self.gcn1(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for i in range(len(self.gcn_hidden)):
                h = self.gcn_hidden[i](h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            h = self.gcn2(h, edge_index)
        out = self.out(h)

        return out, h