import torch.nn as nn
import torch.nn.functional as F

from src.service.implementation.graph_model.meta_module import MetaModule
from src.service.implementation.graph_decoder.meta_linear import MetaLinear



class MetaComposeLinear(MetaModule):
    def __init__(
        self, 
        in_channel: int, 
        hidden_dim: int, 
        out_channel: int
    ) -> None:
        super(MetaComposeLinear, self).__init__()
        
        self.linear1 = MetaLinear(in_channel, hidden_dim)
        self.linear2 = MetaLinear(hidden_dim, out_channel)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x