import polars as pl
from typing import Dict

class Graph:
    def __init__(self,
        nodes: pl.DataFrame,
        edges: pl.DataFrame,
        map_id: Dict[int, int]
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.map_id = map_id
        