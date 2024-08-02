import polars as pl
from typing import Dict
import torch
class Graph:
    
    def __init__(
        self,
        nodes: pl.DataFrame,
        edges: pl.DataFrame,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        map_id: Dict[int, int]    
    ) -> None:
        
        self.nodes = nodes
        self.edges = edges
        self.map_id = map_id
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label