import os
from pathlib import Path
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from thespian.actors import ActorSystem

from src.utils.constants import CONFIG_FILE

FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]


class ApplicationContainer(containers.DeclarativeContainer):

    wiring_config = containers.WiringConfiguration(modules=["src.controller.endpoint"])

    service_config = providers.Configuration()
    service_config.from_yaml(filepath=os.path.join(WORK_DIR, CONFIG_FILE))

    actor_system = providers.Singleton(ActorSystem)