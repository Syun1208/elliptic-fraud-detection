import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class AdaDWLoss(nn.Module):
    
    def __init__(
        self, 
        T: float
    ) -> None:
        super(AdaDWLoss, self).__init__()
        self.T = T
    
    def forward(
        self, 
        loss_trains: List[torch.Tensor], 
        loss_vals: List[torch.Tensor]
    ) -> torch.Tensor:
        
        weights = 1 - loss_trains / loss_vals
        lambda_coeff = torch.exp(weights / self.T) / torch.sum(torch.exp(weights / self.T))
        
        loss_tasks = torch.dot(lambda_coeff, loss_trains)
        
        return loss_tasks