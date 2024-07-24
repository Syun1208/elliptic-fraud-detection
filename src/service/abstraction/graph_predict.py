from abc import ABC, abstractmethod
from src.data_model.score import Score

class Predictor(ABC):
    
    @abstractmethod
    def predict(self) -> Score:
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        pass