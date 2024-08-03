import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import torch
import torch.nn as nn
import torch
import os
import tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool


from src.utils.utils import get_device
from src.utils.timer import time_complexity
from src.service.abstraction.graph_train import Trainer
from src.service.abstraction.data_loader import DataLoader
from torch_geometric.loader import NeighborLoader
from src.service.implementation.graph_loss.adadw_loss import AdaDWLoss
from src.service.implementation.graph_loss.focal_loss import FocalLoss
from src.utils.logger import Logger


FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]

    

class MAXLTrainerImpl(Trainer):
    
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
        self.adadw = AdaDWLoss(T=2)
        self.criterion = FocalLoss(gamma=0)
        
        
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
            
            running_loss_nc = 0.0
            running_loss_lp = 0.0
            val_loss_nc = 0.0
            test_loss_nc = 0.0
            val_loss_lp = 0.0
            test_loss_lp = 0.0
            ap_nc_train = 0.0
            ap_nc_val = 0.0
            ap_nc_test = 0.0
            ap_lp_train = 0.0
            ap_lp_val = 0.0
            ap_lp_test = 0.0
            
            
            for i, batch in enumerate(loader):
                
                self.model.train()
                self.optimizer.zero_grad()
                
                z = self.model.encode(
                        batch.x.to(self.device), 
                        batch.edge_index.to(self.device)
                    ).to(self.device)

                # Node Classification
                out_nc = self.model.node_classification(z)
                loss_nc = self.criterion(
                    out_nc[:batch.batch_size].to(self.device), 
                    batch.y[:batch.batch_size].to(self.device)
                )
                
                # Link Prediction
                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index, 
                    num_nodes=batch.x.shape[0],
                    num_neg_samples=batch.edge_label_index.size(1), method='sparse'
                )

                edge_label_index = torch.cat(
                    [batch.edge_label_index, neg_edge_index],
                    dim=-1,
                ).to(self.device)
                edge_label = torch.cat([
                    batch.edge_label.squeeze(dim=0),
                    batch.edge_label.new_zeros(neg_edge_index.size(1))
                ], dim=0).to(self.device)
                
                out_lp = self.model.link_prediction(z, edge_label_index).view(-1)
                loss_lp = self.criterion(
                    out_lp[:batch.batch_size].to(self.device), 
                    edge_label[:batch.batch_size].to(self.device)
                )
                
                # Append loss of tasks
                running_loss_nc += loss_nc.item()
                running_loss_lp += loss_lp.item()
                
                ap_nc_train += average_precision_score(
                        out_nc[:batch.batch_size].to(self.device).cpu().detach().numpy(), 
                        batch.y[:batch.batch_size].to(self.device).cpu().detach().numpy()[:, 1]
                    )
                
                ap_lp_train += average_precision_score(
                        out_lp[:batch.batch_size].to(self.device).cpu().detach().numpy(), 
                        edge_label[:batch.batch_size].to(self.device)
                    )
            
                loss_nc.backward()
                loss_lp.backward()
                
                self.optimizer.step()
        
            z = self.model.encode(
                    self.data_loader.get_network_torch().x.to(self.device), 
                    self.data_loader.get_network_torch().edge_index.to(self.device)
                )
            
            out_nc = self.model.node_classification(z)
            
            neg_edge_index = negative_sampling(
                    edge_index=self.data_loader.get_network_torch(), num_nodes=batch.x.shape[0],
                    num_neg_samples=self.data_loader.get_network_torch().edge_label_index.size(1), method='sparse')

            edge_label_index = torch.cat(
                [self.data_loader.get_network_torch().edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                self.data_loader.get_network_torch().edge_label.squeeze(dim=0),
                self.data_loader.get_network_torch().edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
                
            out_lp = self.model.link_prediction(z, edge_label_index)
            
            
             ap_nc_val = average_precision_score(
                    self.data_loader.get_network_torch().y[self.data_loader.val_mask].cpu().detach().numpy(), 
                    out_nc[self.data_loader.val_mask].cpu().detach().numpy()[:, 1]
                )

            ap_nc_test = average_precision_score(
                    self.data_loader.get_network_torch().y[self.data_loader.test_mask].cpu().detach().numpy(), 
                    out_nc[self.data_loader.test_mask].cpu().detach().numpy()[:, 1]
                )
            
            ap_lp_val += average_precision_score(
                        out_lp[self.data_loader.val_mask].to(self.device).cpu().detach().numpy(), 
                        edge_label[self.data_loader.val_mask].to(self.device)
                    )
            
            ap_lp_test += average_precision_score(
                        out_lp[self.data_loader.test_mask].to(self.device).cpu().detach().numpy(), 
                        edge_label[self.data_loader.test_mask].to(self.device)
                    )
            
            val_loss_nc = self.criterion(
                self.data_loader.get_network_torch().y[self.data_loader.val_mask].to(self.device),
                out_nc[self.data_loader.val_mask].to(self.device)
            )
            
            test_loss_nc = self.criterion(
                self.data_loader.get_network_torch().y[self.data_loader.test_mask].to(self.device),
                out_nc[self.data_loader.test_mask].to(self.device)
            )
            
            val_loss_lp = self.criterion(
                    out_lp[self.data_loader.val_mask].to(self.device), 
                    edge_label[self.data_loader.val_mask].to(self.device)
                )
            
            test_loss_lp = self.criterion(
                    out_lp[self.data_loader.test_mask].to(self.device), 
                    edge_label[self.data_loader.test_mask].to(self.device)
                )
            
            # Compute and update AdaDW loss
            loss_trains = torch.tensor([loss_nc, loss_lp])
            loss_vals = torch.tensor([val_loss_nc, val_loss_lp])
            
            loss_tasks = self.adadw(loss_trains, loss_vals)
            loss_tasks.backwards()
            
            # Save accuracy and loss to Tensorboard
            self.writer.add_scalars(
                main_tag='Focal Loss - Node Classification', 
                tag_scalar_dict={
                    'Train': running_loss_nc / len(loader),
                    'Validation': val_loss_nc,
                    'Test': test_loss_nc
                }, 
                global_step=epoch
            )
            
            self.writer.add_scalars(
                main_tag='Focal Loss - Link Prediction', 
                tag_scalar_dict={
                    'Train': running_loss_lp / len(loader),
                    'Validation': val_loss_lp,
                    'Test': test_loss_lp
                }, 
                global_step=epoch
            )
            
            self.writer.add_scalars(
                main_tag='AUC-AP - Node Classification',
                tag_scalar_dict={
                    'Train': ap_nc_train / len(loader),
                    'Validation': ap_nc_val,
                    'Test': ap_nc_test
                    },
                global_step=epoch
            )
            
            self.writer.add_scalars(
                main_tag='AUC-AP - Link Prediction',
                tag_scalar_dict={
                    'Train': ap_lp_train / len(loader),
                    'Validation': ap_lp_val,
                    'Test': ap_lp_test
                    },
                global_step=epoch
            )
            
            # Show the computed metrics
            self.logger.info('-' * 5 + 'NODE CLASSIFICATION METRICS' + '-' * 5)
            self.logger.info(f'Loss: {running_loss_nc / len(loader)}')
            self.logger.info(f'AP train: {ap_nc_train / len(loader)}')
            self.logger.info(f'AP val: {ap_nc_val}')
            self.logger.info(f'AP test: {ap_nc_test}')
            self.logger.info(10 * '-')
            
            self.logger.info('-' * 5 + 'LINK PREDICTION METRICS' + '-' * 5)
            self.logger.info(f'Loss: {running_loss_lp / len(loader)}')
            self.logger.info(f'AP train: {ap_lp_train / len(loader)}')
            self.logger.info(f'AP val: {ap_lp_val}')
            self.logger.info(f'AP test: {ap_lp_test}')
            self.logger.info(10 * '-')
          
            
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
