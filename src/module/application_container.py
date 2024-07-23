import os
from pathlib import Path
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from thespian.actors import ActorSystem



FILE = Path(__file__).resolve()
WORK_DIR = FILE.parent.resolve[2]




class ApplicationContainer(containers.DeclarativeContainer):
    pass