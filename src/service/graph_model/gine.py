from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GINConv, GINEConv
from src.service.graph_decoder import *

class GINE(nn.Module):
    def __init__(
            self, 
            num_features: int, 
            edge_dim: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            n_layers: int, 
            dropout_rate: float = 0
            ):
        super().__init__()
        self.num_features = num_features
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers

        if n_layers == 1:
            self.gine1 = GINEConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                    ),
                    edge_dim=edge_dim)
        else:
            self.gine1 = GINEConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                    ),
                    edge_dim=edge_dim)
            
            self.gine_hidden = nn.ModuleList()
            for _ in range(n_layers-2):
                self.gine_hidden.append(GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), 
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()
                        ),
                        edge_dim=edge_dim))
                
            self.gine2 = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                    ),
                    edge_dim=edge_dim)
        
        self.out = DecoderLinear(embedding_dim)
        
    def forward(self, x, edge_index, edge_features):
        h = self.gine1(x, edge_index, edge_features)

        for layer in self.gine_hidden:
            h = layer(h, edge_index, edge_features)

        h = self.gine2(h, edge_index, edge_features)
        out = self.out(h)

        return out, h