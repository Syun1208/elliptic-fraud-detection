import polars as pl
from typing import List


class Score:
    
    def __init__(
        self,
        ap_scores: List[str],
        ra_scores: List[str]
    ) -> None:
        
        self.ap_scores = ap_scores
        self.ra_scores = ra_scores
    

    
    def to_frame(self) -> pl.DataFrame:
        
        return pl.DataFrame({
            'average_precision': self.ap_scores,
            'ra_scores': self.ra_scores
        })