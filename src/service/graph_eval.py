from abc import ABC, abstractmethod
import polars as pl
from pathlib import Path
import os

from src.utils.timer import time_complexity
from src.service.graph_test import Tester
from src.utils.logger import Logger


FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]

class Evaluator(ABC):
    
    @abstractmethod
    def evaluate(self) -> pl.DataFrame:
        pass
    
    
    @abstractmethod
    def to_results(self) -> None:
        pass
    
    
    
class EvaluatorImpl(Evaluator):
    
    def __init__(
        self,
        tester: Tester,
        logger: Logger,
        path_results: str
    ) -> None:
        
        self.tester = tester
        self.logger = logger.get_tracking(__name__)
        self.path_results = path_results
    
    @time_complexity(name_process='PHASE EVALUATE')
    def evaluate(self) -> pl.DataFrame:
        
        self.predict = self.tester.predict()
        self.to_results(
            os.path.join(WORK_DIR, self.path_results)
        )
    
    
    def to_results(self, path: str) -> None:
        
        self.predict.to_frame().write_csv(path) 
        self.logger.info(f'Save results at: {path}')