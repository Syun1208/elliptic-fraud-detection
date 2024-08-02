import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.service.implementation.graph_decoder.decoder_linear import DecoderLinear

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
            for _ in range(n_layers - 2): 
                self.gcn_hidden.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn2 = GCNConv(hidden_dim, embedding_dim)
            
        self.out = DecoderLinear(self.embedding_dim, self.output_dim)
        
    def encode(
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
            h = self.gcn2(h, edge_index)
        
        return h
    
    def node_classification(
        self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        
        output = self.out(z)
        
        return output
        
        
    def link_prediction(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor
        ) -> torch.Tensor:
        
        output = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        
        return output
        
