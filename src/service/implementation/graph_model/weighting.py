import torch.nn.functional as F

from src.service.implementation.graph_model.meta_module import MetaModule
from src.service.implementation.graph_decoder.meta_linear import MetaLinear



class Weight(MetaModule):
    def __init__(self, in_channel, hidden, out_channel):
        super(Weight, self).__init__()
        self.linear1 = MetaLinear(in_channel, hidden)
        self.linear2 = MetaLinear(hidden, out_channel)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.sigmoid(x)
        return x