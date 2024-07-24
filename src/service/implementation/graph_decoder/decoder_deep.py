import torch.nn as nn
import torch.nn.functional as F
    
class DecoderDeep(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            n_layers: int,
            hidden_dim: int,
            output_dim: int=2
            ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, 
        embedding,
        normalize=False
    ):
        if normalize:
            embedding = self.layer_norm(embedding)
            
        h = self.layer1(embedding)
        h = F.relu(h)
        for layer in self.layers:
            h = layer(h)
            h = F.relu(h)
        h = self.layer2(h)
        h = self.softmax(h)
        return h