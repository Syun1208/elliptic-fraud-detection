import torch.nn as nn
import torch.nn.functional as F

from src.service.implementation.graph_model.meta_module import MetaModule


class MetaLinear(MetaModule):
    def __init__(
        self,
        num_features: int,
        embedding_dim: int
    ) -> None:
        super().__init__()
        ignore = nn.Linear(num_features, embedding_dim)

        self.register_buffer('weight', self.to_var(ignore.weight, requires_grad=True))
        self.register_buffer('bias', self.to_var(ignore.bias, requires_grad=True))
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]