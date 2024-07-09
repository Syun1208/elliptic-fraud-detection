import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import *




class KANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(KANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        
        #This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik, djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        
        y = y.view(outshape)
        return y




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