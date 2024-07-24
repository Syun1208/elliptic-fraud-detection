import torch.nn as nn
import torch.nn.functional as F

class DecoderLinear(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            output_dim: int=2
            ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.layer1 = nn.Linear(embedding_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, 
        embedding, 
        normalize=False
    ):
        if normalize:
            embedding = self.layer_norm(embedding)
        h = self.layer1(embedding)
        h = self.softmax(h)
        return h