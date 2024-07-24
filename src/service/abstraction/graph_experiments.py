from abc import ABC, abstractmethod


class Experiments(ABC):
    
    @abstractmethod
    def run(self) -> None:
        pass