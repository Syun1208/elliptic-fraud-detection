from abc import ABC, abstractmethod
import polars as pl


class Evaluator(ABC):
    
    @abstractmethod
    def evaluate(self) -> pl.DataFrame:
        pass
    
    
    @abstractmethod
    def to_results(self) -> None:
        pass