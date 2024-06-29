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
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool


from src.utils.utils import get_device
from src.utils.timer import time_complexity
from src.service.data_loader import DataLoader
from torch_geometric.loader import NeighborLoader
from src.utils.visualizer import plot_confusion_matrix
from src.utils.logger import Logger


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
        path_model: str
    ) -> None:
        
        self.model = model
        self.data_loader = data_loader.load()
        self.logger = logger.get_tracking(__name__)
        self.epochs = epochs
        self.lr = lr
        self.device_id = device_id
        self.batch_size = batch_size
        self.path_model = path_model
        self.path_logs_tensorboard = os.path.join(WORK_DIR, path_logs_tensorboard)
        if not os.path.exists(self.path_logs_tensorboard):
            os.makedirs(self.path_logs_tensorboard)
            
        self.writer = SummaryWriter(self.path_logs_tensorboard)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        self.device = get_device(self.device_id)
        
        
        self.logger.info(f'MODEL: \n {self.model}')
        self.logger.info(f'HYPERPARAMETERS: \n \
                         epochs: {self.epochs} \n \
                         learning-rate: {self.lr} \n \
                         batch-size: {self.batch_size}')
        self.logger.info(f'DEVICE: {self.device}')
    
    @time_complexity(name_process='PHASE TRAIN')
    def fit(self) -> None:

        try:
            loader = NeighborLoader(
                data=self.data_loader.get_network_torch(), 
                num_neighbors=[-1]*self.model.n_layers, 
                input_nodes=self.data_loader.train_mask, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=Pool()._processes
            )
        except:
            loader = NeighborLoader(
                data=self.data_loader.get_network_torch(), 
                num_neighbors=[-1]* self.model.n_layers, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=Pool()._processes
            )

        
        self.model.to(self.device)

        for epoch in tqdm.tqdm(range(self.epochs), colour='green', desc='Training graph model'):
            
            running_loss = 0.0
            
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

                loss.backward()
                self.optimizer.step()
       
            # Save accuracy and loss to Tensorboard
            self.writer.add_scalar(
                tag='Loss/train', 
                scalar_value=running_loss, 
                global_step=epoch
            )
            
            self.logger.info(f'Loss: {running_loss / len(loader)}')
          
            
        self.logger.info('DONE PHASE TRAIN !')
        
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