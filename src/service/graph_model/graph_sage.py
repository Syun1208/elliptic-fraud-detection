from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GINConv, GINEConv
from src.service.graph_decoder import *


class GraphSAGE(nn.Module): #Neighbourhood sampling only in training step (via DataLoader)
    def __init__(
            self,
            num_features: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            n_layers: int, 
            dropout_rate: float = 0, 
            sage_aggr: str='mean'
            ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers
        self.sage_aggr = sage_aggr

        if n_layers == 1:
            self.sage1 = SAGEConv(num_features, embedding_dim, aggr=sage_aggr)
        else:
            self.sage1 = SAGEConv(num_features, hidden_dim, aggr=sage_aggr)
            self.sage_hidden = nn.ModuleList()
            for _ in range(n_layers-2):
                self.sage_hidden.append(SAGEConv(hidden_dim, hidden_dim, aggr=sage_aggr))
            
            self.sage2 = SAGEConv(hidden_dim, embedding_dim, aggr=sage_aggr)

        self.out = DecoderLinear(embedding_dim, output_dim)
        
    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.sage_hidden:
                h = layer(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            h = self.sage2(h, edge_index)
        out = self.out(h)
        
        return out, h