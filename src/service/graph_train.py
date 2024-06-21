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
        self.data_loader = data_loader
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
    
    
    
    @time_complexity(name_process='PHASE TRAIN')
    def fit(self) -> None:
        
        
        try:
            loader = NeighborLoader(
                self.data_loader, 
                num_neighbors= [-1]*self.model.n_layers, 
                input_nodes=self.data_loader.train_mask, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=Pool()._processes
            )
        except:
            loader = NeighborLoader(
                self.data_loader, 
                num_neighbors= [-1]* self.model.n_layers, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=Pool()._processes
            )
            
        device = torch.device(f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        
        for epoch in tqdm.tqdm(self.epochs, colour='green', desc='Training graph model'):
            
            running_loss = 0.0
            accuracy = 0
            ap_score = 0
            
            self.model.train()
            
            for i, batch in enumerate(loader):
                
                self.optimizer.zero_grad()
                
                out, h = self.model(batch.x, batch.edge_index.to(device))
                
                y_hat = out[:batch.batch_size].to(device)
                y = batch.y[:batch.batch_size].to(device)
                
                loss = self.criterion(y_hat, y)
                accuracy += torch.sum(y_hat == y)
                 
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.items()
                
                if i % 1000 == 999:   
                             
                    steps = epoch * len(loader) + i 
                    batch = i * self.batch_size 
                    
                    # Save accuracy and loss to Tensorboard
                    self.writer.add_scalar(tag='Loss/train', scalar_value=running_loss / batch, global_step=steps)
                    self.writer.add_scalar(tag='Accuracy/train', scalar_value=accuracy / batch, global_step=steps)
                 
            ap_score = average_precision_score(
                    y.cpu().detach().numpy(), 
                    y_hat.cpu().detach().numpy()[:,1]
            )  
            
            ra_score = roc_auc_score(
                y.cpu().detach().numpy(), 
                y_hat.cpu().detach().numpy()[:,1]
            )
            
            self.writer.add_scalar(
                tag='AveragePrecision/train', 
                scalar_value=ap_score, 
                global_step=epoch
            )
            
            self.writer.add_scalar(
                tag='AUCROC/train',
                scalar_value=ra_score,
                global_step=epoch
            )
            
            self.writer.add_figure(
                "ConfusionMatrix/train", 
                plot_confusion_matrix(
                    y_true=y.cpu().detach().numpy(), 
                    y_pred=y_hat.cpu().detach().numpy()[:, 1]
                ), 
                epoch
            )
            
        self.logger.info('DONE PHASE TRAIN !')
        
        # Save the trained model
        self.to_model(self.path_model)
        
    
    def to_model(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)