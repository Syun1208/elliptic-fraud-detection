import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils import *

from src.service.implementation.graph_decoder.kan import KANLayer



class KanGNN(nn.Module):
    def __init__(
        self, 
        n_features: int, 
        hidden_dim: int, 
        output_dim: int, 
        grid_dim: int, 
        n_layers: int, 
        use_bias: bool = False
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.lin_in = nn.Linear(n_features, hidden_dim, bias=use_bias)
        #self.lin_in = KANLayer(n_features, hidden_dim, grid_feat, addbias=use_bias)
        self.lins = torch.nn.ModuleList()
        for i in range(n_layers):
            self.lins.append(KANLayer(hidden_dim, hidden_dim, grid_dim, addbias=use_bias))
        self.lins.append(nn.Linear(hidden_dim, output_dim, bias=False))
        #self.lins.append(KANLayer(hidden_dim, output_dim, grid_feat, addbias=False))

        # self.lins = torch.nn.ModuleList()
        # self.lins.append(nn.Linear(n_features, hidden_dim, bias=use_bias))
        # for i in range(n_layers):
        #     self.lins.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
        # self.lins.append(nn.Linear(hidden_dim, output_dim, bias=use_bias))

    
    def forward(self, x, adj):
        x = self.lin_in(x)
        #x = self.lin_in(spmm(adj, x))
        for layer in self.lins[:self.n_layers-1]:
            x = layer(spmm(adj, x))
            #x = layer(x)
        x = self.lins[-1](x)
            
        return x.log_softmax(dim=-1)