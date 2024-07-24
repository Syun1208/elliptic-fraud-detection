from abc import abstractmethod, ABC


class Trainer(ABC):

    @abstractmethod
    def fit(self) -> None:
        pass
    
    @abstractmethod
    def to_model(self) -> None:
        pass