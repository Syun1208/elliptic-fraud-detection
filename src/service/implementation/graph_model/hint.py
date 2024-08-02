import torch
from typing import List, Tuple
from torch_geometric.nn.conv import GATConv

from src.service.implementation.graph_model.meta_module import MetaModule
from src.service.implementation.graph_model.gat_tuning import GATTuning


class MetaGAT(MetaModule):
    
    def __init__(
        self, 
        in_channel: int,
        out_channel: int
    ) -> None:
        ignore = GATConv(in_channel, out_channel)
        self.register_buffer('weight', self.to_var(ignore.lin_src.weight, requires_grad=True))
        self.register_buffer('bias', self.to_var(ignore.bias, requires_grad=True))
        self.register_buffer('att_src', self.to_var(ignore.att_src, requires_grad=True))
        self.register_buffer('att_dst', self.to_var(ignore.att_dst, requires_grad=True))
        
        self.gat = GATTuning(ignore.weight.shape[1], ignore.weight.shape[0], self.weight, self.att, self.bias)

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        
        return self.gat(x, edge_index)

    def named_leaves(self) -> List[Tuple[str, torch.Tensor]]:
        return [('weight', self.weight), ('att', self.att), ('bias', self.bias)]