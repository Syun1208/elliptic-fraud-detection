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
        
        loss_train = loss_trains[1] + loss_trains[0]
        loss_val = loss_vals[1] + loss_vals[0]
        
        weights = 1 - loss_train / loss_val
        lambda_coeff = torch.exp(weights / self.T) / torch.sum(torch.exp(weights / self.T))
        
        loss_tasks = lambda_coeff * loss_train
        
        return loss_tasks