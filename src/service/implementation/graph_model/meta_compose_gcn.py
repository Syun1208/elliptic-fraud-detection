import torch.nn as nn
import torch
import torch.nn.functional as F

from src.service.implementation.graph_model.meta_module import MetaModule
from src.service.implementation.graph_model.meta_gcn import MetaGCN



class MetaComposeGCN(MetaModule):
    def __init__(
        self,
        num_features: int, 
        hidden_dim: int, 
        embedding_dim: int, 
        output_dim: int, 
        n_layers: int, 
        dropout_rate: float
    ) -> None:
        super(MetaComposeGCN, self).__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn_hidden = nn.ModuleList()
        self.n_layers = n_layers
        
        if self.n_layers == 1:
            self.gcn1 = MetaGCN(num_features, embedding_dim)
        else:
            self.gcn1 = MetaGCN(num_features, hidden_dim)
            for _ in range(n_layers-2): 
                self.gcn_hidden.append(MetaGCN(hidden_dim, hidden_dim))
            self.gcn2 = MetaGCN(hidden_dim, embedding_dim)
            
            
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        if self.n_layers > 1:
            for layer in self.gcn_hidden:
                h = layer(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            x = self.gcn2(h, edge_index)
            
        return x