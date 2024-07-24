import polars as pl
from pathlib import Path
import os

from src.utils.timer import time_complexity
from src.service.abstraction.graph_eval import Evaluator
from src.service.abstraction.graph_predict import Predictor
from src.utils.logger import Logger


FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]
    
    
    
class EvaluatorImpl(Evaluator):
    
    def __init__(
        self,
        predictor: Predictor,
        logger: Logger,
        path_results: str
    ) -> None:
        
        self.predictor = predictor
        self.logger = logger.get_tracking(__name__)
        self.path_results = path_results
    
    @time_complexity(name_process='PHASE EVALUATE')
    def evaluate(self) -> pl.DataFrame:
        
        self.predict = self.predictor.predict()
        self.to_results(
            os.path.join(WORK_DIR, self.path_results)
        )
    
    
    def to_results(self, path: str) -> None:
        
        self.predict.to_frame().write_csv(path) 
        self.logger.info(f'Save results at: {path}')