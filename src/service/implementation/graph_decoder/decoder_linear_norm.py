import torch.nn as nn
import torch.nn.functional as F

    
class DecoderLinearNorm(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            output_dim: int=2
            ):
        super().__init__()
        self.normalise = nn.LayerNorm(embedding_dim)
        self.layer1 = nn.Linear(embedding_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embedding):
        h = self.normalise(embedding)
        h = self.layer1(h)
        h = self.softmax(h)
        return h