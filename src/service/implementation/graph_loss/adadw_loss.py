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
        
        num_tasks = len(loss_trains)
        weights = []
        lambda_coeffs = []
        
        for i in range(num_tasks):
            
            weight = 1 - loss_trains[i] / loss_vals[i]
            lambda_coeff = torch.exp(weight / self.T)
            
            lambda_coeffs.append(lambda_coeff)
            weights.append(weights)
            
        lambda_coeffs = [i / sum(lambda_coeffs) for i in lambda_coeffs]
        loss_tasks = sum([loss_trains[i] * lambda_coeffs[i] for i in range(num_tasks)])
        
        return loss_tasks