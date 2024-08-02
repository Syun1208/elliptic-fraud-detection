import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from src.service.implementation.graph_decoder.decoder_linear import DecoderLinear


class GAT(nn.Module):
    def __init__(
            self, 
            num_features: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            n_layers: int, 
            heads: int = 1, 
            dropout_rate: float = 0
            ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers
        self.heads = heads

        if n_layers == 1:
            self.gat1 = GATv2Conv(num_features, embedding_dim, heads=heads, concat=False)
        else:
            self.gat1 = GATv2Conv(num_features, hidden_dim, heads=heads)
            self.gat_hidden = nn.ModuleList()
            for _ in range(n_layers-2):
                self.gat_hidden.append(GATv2Conv(heads*hidden_dim, hidden_dim, heads=heads))
            self.gat2 = GATv2Conv(heads*hidden_dim, embedding_dim, heads=heads, concat=False)

        self.out = DecoderLinear(embedding_dim)

    def forward(self, x, edge_index, edge_features=None):
        h = self.gat1(x, edge_index, edge_attr=edge_features)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.gat_hidden:
                h = layer(h, edge_index, edge_attr=edge_features)
                h = F.relu(h)
                h = self.dropout(h)
            
            h = self.gat2(h, edge_index, edge_attr=edge_features)
        out = self.out(h)
        
        return out, h