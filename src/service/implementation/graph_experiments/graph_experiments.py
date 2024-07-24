import traceback

from src.utils.logger import Logger
from src.service.abstraction.graph_experiments import Experiments
from src.utils.timer import time_complexity
from service.implementation.graph_train.trainer import Trainer
from src.service.abstraction.graph_predict import Predictor
from service.implementation.graph_eval.evaluator import Evaluator

    

class ExperimentsImpl(Experiments):
    
    def __init__(
        self,
        trainer: Trainer,
        predictor: Predictor,
        evaluator: Evaluator,
        logger: Logger
    ) -> None:
    
        self.logger = logger.get_tracking(__name__)
        self.trainer = trainer
        self.predictor = predictor
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
                self.predictor.predict()
        
        except Exception as e:
            
            self.logger.error(e)
            self.logger.error(traceback.format_exc())