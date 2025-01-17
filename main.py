import warnings
import argparse
from dependency_injector.wiring import Provide, inject
from threadpoolctl import threadpool_limits, threadpool_info

from src.utils.debugger import pretty_errors
from src.module.application_container import ApplicationContainer
from src.service.implementation.graph_experiments.graph_experiments import ExperimentsImpl

warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, help='train/eval/test', default='train')
    return parser.parse_args()


def create_container(environment: str):
    container = ApplicationContainer()
    container.wire(modules=[environment])
    
    
@inject
def run(experiments: ExperimentsImpl = Provide[ApplicationContainer.experiments], phase: str = 'train') -> None:
    experiments.run(phase=phase)
      
    
create_container(environment=__name__)


if __name__ == '__main__':
    
    args = parse_arg()
    run(phase=args.phase)