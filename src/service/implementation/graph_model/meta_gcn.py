import torch
from typing import List, Tuple
from torch_geometric.nn import GCNConv
from src.service.implementation.graph_model.gcn_tuning import GCNTuning
from src.service.implementation.graph_model.meta_module import MetaModule


class MetaGCN(MetaModule):
    def __init__(
        self, 
        in_channel: int,
        out_channel: int
    ) -> None:
        super(MetaGCN, self).__init__()
        ignore = GCNConv(in_channel, out_channel)
        self.register_buffer('weight', self.to_var(ignore.lin.weight, requires_grad=True))
        self.register_buffer('bias', self.to_var(ignore.bias, requires_grad=True))
        self.gcn = GCNTuning(self.weight.shape[1], self.weight.shape[0], self.weight, self.bias)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        
        return self.gcn(x, edge_index)

    def named_leaves(self) -> List[Tuple[str, torch.Tensor]]:
        return [('weight', self.weight), ('bias', self.bias)]