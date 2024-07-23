from abc import ABC, abstractmethod
import traceback

from src.utils.logger import Logger
from src.utils.timer import time_complexity
from src.service.graph_train import Trainer
from src.service.graph_test import Tester
from src.service.graph_eval import Evaluator


class Experiments(ABC):
    
    @abstractmethod
    def run(self) -> None:
        pass
    

class ExperimentsImpl(Experiments):
    
    def __init__(
        self,
        trainer: Trainer,
        tester: Tester,
        evaluator: Evaluator,
        logger: Logger
    ) -> None:
    
        self.logger = logger.get_tracking(__name__)
        self.trainer = trainer
        self.tester = tester
        self.evaluator = evaluator
        
        
    
    @time_complexity(name_process='PHASE EXPERIMENT')
    def run(self, phase: str):
        
        try:
            
            if phase == 'train':
                
                self.logger.info('Start training !')
                self.trainer.fit()
            
            elif phase == 'eval':
                
                self.logger.info('Start evaluating !')
                self.evaluator.evaluate()
            
            else:
                
                self.logger.info('Start predicting !')
                self.tester.predict()
        
        except Exception as e:
            
            self.logger.error(e)
            self.logger.error(traceback.format_exc())