from abc import ABC, abstractmethod
from src.data_model.network import DataNetWork

class DataLoader(ABC):
    
    @abstractmethod
    def load(self) -> DataNetWork:
        pass