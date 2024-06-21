import warnings
from dependency_injector.wiring import Provide, inject

from src.utils.debugger import pretty_errors
from src.module.application_container import ApplicationContainer
from src.service.graph_experiments import ExperimentsImpl


warnings.filterwarnings('ignore')

def create_container(environment: str):
    container = ApplicationContainer()
    container.wire(modules=[environment])
    
    
@inject
def train(experiments: ExperimentsImpl = Provide[ApplicationContainer.experiments]) -> None:
    experiments.run(phase='train')
    
@inject
def test(experiments: ExperimentsImpl = Provide[ApplicationContainer.experiments]) -> None:
    experiments.run(phase='test')
    
@inject
def evaluate(experiments: ExperimentsImpl = Provide[ApplicationContainer.experiments]) -> None:
    experiments.run(phase='eval')
    
    
    
create_container(environment=__name__)


if __name__ == '__main__':
    train()