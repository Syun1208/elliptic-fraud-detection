from torch import nn
import torch

from src.service.implementation.graph_model.gcn import GCN, DecoderLinear
from src.service.implementation.graph_model.egcl import EGCL


class EGNN(nn.Module):
    def __init__(
        self, 
        num_features: int, 
        hidden_dim: int, 
        embedding_dim: int, 
        output_dim: int= 1, 
        n_layers: int = 3, 
        dropout_rate: float = 0,
        in_edge_nf: int =0, 
        act_fn: torch.nn.functional = nn.SiLU(), 
        residual: bool = True,
        attention: bool = False, 
        normalize: bool = False, 
        tanh: bool = False
    ) -> None:
        '''
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        
        self.gcn = GCN(
            num_features=num_features, 
            hidden_dim=hidden_dim, 
            embedding_dim=embedding_dim, 
            output_dim=output_dim, 
            n_layers=n_layers, 
            dropout_rate=dropout_rate
        )
        
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, EGCL(
                    hidden_dim, 
                    hidden_dim, 
                    hidden_dim, 
                    edges_in_d=2,                        
                    act_fn=act_fn, 
                    residual=residual, 
                    attention=attention,
                    normalize=normalize, 
                    tanh=tanh
                )
            )

    def node_classification(
        self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        
        output = self.gcn.node_classification(z)
        
        return output
        
        
    def link_prediction(
        self,
        z: torch.Tensor,
        edge_label_index: torch.Tensor
        ) -> torch.Tensor:

        output = self.gcn.link_prediction(z, edge_label_index)
        
        
        return output
    
    
    def encode(
        self, 
        x, 
        edge_index, 
        edge_attr=None
    ) -> None:
        h = self.gcn.encode(x, edge_index)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edge_index, x, edge_attr=edge_attr)
   
        return h