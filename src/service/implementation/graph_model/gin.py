import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from src.service.implementation.graph_decoder.decoder_linear import DecoderLinear


class GIN(nn.Module):
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
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers

        if n_layers == 1:
            self.gin1 = GINConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                    ))
            
        else:
            self.gin1 = GINConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                    ))
            
            self.gin_hidden = nn.ModuleList()
            for _ in range(n_layers-2):
                self.gin_hidden.append(GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), 
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()
                        )))
            
            self.gin2 = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                    ))

        self.out = DecoderLinear(embedding_dim)
    
    def forward(self, x, edge_index):
        h = self.gin1(x, edge_index)

        if self.n_layers > 1:
            for layer in self.gin_hidden:
                h = layer(h, edge_index)

            h = self.gin2(h, edge_index)
        out = self.out(h)

        return out, h