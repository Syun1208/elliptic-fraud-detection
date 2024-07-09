import os
from pathlib import Path
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from thespian.actors import ActorSystem

from src.utils.constants import CONFIG_FILE
from src.utils.debugger import pretty_errors
from src.service.data_loader import DataLoader, EllipticLoader
from src.service.graph_model.gcn import GCN
from src.service.graph_model.gat import GAT
from src.service.graph_model.gin import GIN
from src.service.graph_model.graph_sage import GraphSAGE
from src.service.graph_model.graph_kan import KanGNN
from src.service.graph_train import Trainer, TrainerImpl
from src.service.graph_test import Tester, TesterImpl
from src.service.graph_eval import Evaluator, EvaluatorImpl
from src.service.graph_experiments import Experiments, ExperimentsImpl
from src.utils.logger import Logger

FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]


class ApplicationContainer(containers.DeclarativeContainer):

    wiring_config = containers.WiringConfiguration(modules=["src.service.graph_experiments"])

    service_config = providers.Configuration()
    service_config.from_yaml(filepath=os.path.join(WORK_DIR, CONFIG_FILE))
    actor_system = providers.Singleton(ActorSystem)
    
    
    logger = providers.Singleton(
        Logger,
        log_dir=service_config.logger.log_dir,
        log_clear_days=service_config.logger.log_clear_days
    )
    
    
    elliptic_loader = providers.AbstractSingleton(DataLoader)
    elliptic_loader.override(
        providers.Singleton(
            EllipticLoader,
            path_features=service_config.data.path_features,
            path_edgelist=service_config.data.path_edgelist,
            path_classes=service_config.data.path_classes
        )
    )
    
    gat = providers.Singleton(
        GAT,
        num_features=service_config.gat.n_features,
        hidden_dim=service_config.gat.hidden_dim,
        embedding_dim=service_config.gat.embedding_dim,
        output_dim=service_config.gat.output_dim,
        n_layers=service_config.gat.n_layers,
        heads=service_config.gat.heads,
        dropout_rate=service_config.gat.dropout_rate
    )
    
    gcn = providers.Singleton(
        GCN,
        num_features=service_config.gat.n_features,
        hidden_dim=service_config.gat.hidden_dim,
        embedding_dim=service_config.gat.embedding_dim,
        output_dim=service_config.gat.output_dim,
        n_layers=service_config.gat.n_layers,
        dropout_rate=service_config.gat.dropout_rate
    )

    gin = providers.Singleton(
        GIN,
        num_features=service_config.gat.n_features,
        hidden_dim=service_config.gat.hidden_dim,
        embedding_dim=service_config.gat.embedding_dim,
        output_dim=service_config.gat.output_dim,
        n_layers=service_config.gat.n_layers,
        dropout_rate=service_config.gat.dropout_rate
    )

    graph_sage = providers.Singleton(
        GraphSAGE,
        num_features=service_config.gat.n_features,
        hidden_dim=service_config.gat.hidden_dim,
        embedding_dim=service_config.gat.embedding_dim,
        output_dim=service_config.gat.output_dim,
        n_layers=service_config.gat.n_layers,
        dropout_rate=service_config.gat.dropout_rate
    )
    
    graph_kan = providers.Singleton(
        KanGNN,
        n_features=service_config.kan_gnn.n_features,
        hidden_dim=service_config.kan_gnn.hidden_dim, 
        output_dim=service_config.kan_gnn.output_dim, 
        grid_dim=service_config.kan_gnn.grid_dim,
        n_layers=service_config.kan_gnn.n_layers,
        use_bias=False,
    )
    
    
    trainer = providers.AbstractSingleton(Trainer)
    trainer.override(
        providers.Singleton(
            TrainerImpl,
            model=graph_kan,
            data_loader=elliptic_loader,
            logger=logger, 
            epochs=service_config.kan_gnn.epochs,
            lr=service_config.kan_gnn.lr,
            batch_size=service_config.kan_gnn.batch_size,
            device_id=service_config.device_id,
            path_logs_tensorboard=service_config.tensorboard.log_dir,
            path_model=service_config.kan_gnn.path_model
        )
    )
    

    tester = providers.AbstractSingleton(Tester)
    tester.override(
        providers.Singleton(
            TesterImpl,
            model=graph_kan,
            data_loader=elliptic_loader,
            logger=logger,
            device_id=service_config.device_id,
            batch_size=service_config.kan_gnn.batch_size,
            path_model=service_config.kan_gnn.path_model,
            n_random_samples=service_config.kan_gnn.test.n_test
        )
    )
    
    evaluator = providers.AbstractSingleton(Evaluator)
    evaluator.override(
        providers.Singleton(
            EvaluatorImpl,
            tester=tester,
            logger=logger,
            path_results=service_config.gat.test.path_results
        )
    )
    
    experiments = providers.AbstractSingleton(Experiments)
    experiments.override(
        providers.Singleton(
            ExperimentsImpl,
            trainer=trainer,
            tester=tester,
            evaluator=evaluator,
            logger=logger
        )
    )