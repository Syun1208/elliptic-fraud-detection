import warnings

warnings.filterwarnings('ignore')

from abc import abstractmethod, ABC
from pathlib import Path
import torch
import torch.nn as nn
import torch
import os
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from multiprocessing import Pool


from src.utils.utils import get_device
from src.utils.timer import time_complexity
from src.service.data_loader import DataLoader
from torch_geometric.loader import NeighborLoader
from src.utils.visualizer import plot_confusion_matrix
from src.utils.logger import Logger
from src.service.graph_loss.focal_loss import FocalLoss



import mlflow
from mlflow.tracking import MlflowClient
from src.utils.mlflow_uitls import log_model
from src.utils.mlflow_uitls import set_or_create_experiment
FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]


class Trainer(ABC):

    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def to_model(self):
        pass
    
    
class TrainerImpl(Trainer):
    def __init__(
        self,
        model: torch.nn,
        data_loader: DataLoader,
        logger: Logger, 
        epochs: int,
        lr: float,
        batch_size: int,
        device_id: int,
        path_logs_tensorboard: str,
        path_model: str) -> None:
        
        
        self.model = model
        self.data_loader = data_loader.load()
        self.logger = logger.get_tracking(__name__)
        self.epochs = epochs
        self.lr = lr
        self.device_id = device_id
        self.batch_size = batch_size
        self.path_model = path_model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = FocalLoss(gamma=0)
        self.device = get_device(self.device_id)
        
        ########################## MLFLOW ################################
        self.params = {
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "loss_function": self.criterion.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "device": self.device
        }
        self.experiment_name = "Fraud Detection"
        self.run_name = "training_fraud_detection"
        self.model_name = "GCKanV2"
        self.artifact_path = "artifacts"
        ##################################################################
        
        self.logger.info(f'MODEL: \n {self.model}')
        self.logger.info(f'HYPERPARAMETERS: \n \
                         epochs: {self.epochs} \n \
                         learning-rate: {self.lr} \n \
                         batch-size: {self.batch_size}')
        self.logger.info(f'DEVICE: {self.device}')
        
    def fit(self) -> None:

        loader = NeighborLoader(
            data=self.data_loader.get_network_torch(), 
            num_neighbors=[-1]*self.model.n_layers, 
            input_nodes=self.data_loader.train_mask, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=Pool()._processes
        )
    

        
        self.model.to(self.device)
        
        experiment_id = set_or_create_experiment(experiment_name=self.experiment_name)
        with mlflow.start_run(run_name=self.run_name) as run:            
            log_model(self.model, self.params)
            mlflow.pytorch.log_model(self.model, artifact_path = self.artifact_path, registered_model_name=self.model_name)
            client = MlflowClient()
            model_version = client.get_latest_versions(name=self.model_name, stages=["None"])[0].version
            client.update_model_version(name= self.model_name,
                                        version = model_version,
                                        description=f"Run id: {run.info.run_id}\nVersion: {model_version}\nRegistered model name: {self.model_name}")
            for epoch in tqdm.tqdm(range(self.epochs), colour='green', desc='Training graph model'):
                
                running_loss = 0.0
                ap_score = 0.0
                ap_score_val = 0.0
                ap_score_test = 0.0
                self.model.train()
                
                for i, batch in enumerate(loader):
                    
                    self.optimizer.zero_grad()

                    out, h = self.model(
                        batch.x.to(self.device), 
                        batch.edge_index.to(self.device)
                    )
                    
                    y_hat = out[:batch.batch_size].to(self.device)
                    y = batch.y[:batch.batch_size].to(self.device)
                    
                    loss = self.criterion(y_hat, y)
                    running_loss += loss.item()
                    ap_score += average_precision_score(
                        y.cpu().detach().numpy(), 
                        y_hat.cpu().detach().numpy()[:, 1]
                    )
                    
                    loss.backward()
                    self.optimizer.step()
                
                out, h = self.model(
                        self.data_loader.get_network_torch().x.to(self.device), 
                        self.data_loader.get_network_torch().edge_index.to(self.device)
                    )
                
                ap_score_val = average_precision_score(
                        self.data_loader.get_network_torch().y[self.data_loader.val_mask].cpu().detach().numpy(), 
                        out[self.data_loader.val_mask].cpu().detach().numpy()[:, 1]
                    )

                ap_score_test = average_precision_score(
                        self.data_loader.get_network_torch().y[self.data_loader.test_mask].cpu().detach().numpy(), 
                        out[self.data_loader.test_mask].cpu().detach().numpy()[:, 1]
                    )
                ###########################################################################################
                
                mlflow.log_metric("loss/train",value=running_loss / len(loader), step=epoch)
                mlflow.log_metric("AP/train",value=ap_score / len(loader), step=epoch)
                mlflow.log_metric("AP/val",value=ap_score_val, step=epoch)
                mlflow.log_metric("AP/test",value=ap_score_test, step=epoch)
                
                ############################################################################################
                self.logger.info(f'Loss: {running_loss / len(loader)}')
                self.logger.info(f'AP train: {ap_score / len(loader)}')
                self.logger.info(f'AP val: {ap_score_val}')
                self.logger.info(f'AP test: {ap_score_test}')

                # self.lr_schedule.step()
                
            self.logger.info('DONE PHASE TRAIN !')
            
            mlflow.pytorch.log_model(self.model, "model")
        # Save the trained model
        self.to_model(
            os.path.join(WORK_DIR, self.path_model)
        )
        
    
    def to_model(self, path: str) -> None:
        
        os.makedirs(
            '/'.join(os.path.splitext(path)[0].split('/')[:-1]), 
            exist_ok=True
        )
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)

        self.logger.info(f'SAVE YOUR MODEL AT: {path}')