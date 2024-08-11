import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
from pathlib import Path
import torch
import torch_geometric.transforms as T
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
        self.criterion = FocalLoss(gamma=5, alpha=[0.25, 0.75])
        self.criterion_lp = nn.BCEWithLogitsLoss()
        self.transform = T.Compose([
            T.RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=False,
                neg_sampling_ratio=1.0,
            )
        ])
        
        self.device = get_device(self.device_id)
        
        
        self.logger.info(f'MODEL: \n {self.model}')
        self.logger.info(f'HYPERPARAMETERS: \n \
                         epochs: {self.epochs} \n \
                         learning-rate: {self.lr} \n \
                         batch-size: {self.batch_size}')
        self.logger.info(f'DEVICE: {self.device}')
    
    @time_complexity(name_process='PHASE TRAIN')
    def fit(self) -> None: 

        train, val, test = self.transform(self.data_loader.get_network_torch())
        
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
            

            # PHASE: TRAIN
            self.model.train()
            self.optimizer.zero_grad()
            
            z = self.model.encode(
                    self.data_loader.get_network_torch().x[self.data_loader.get_network_torch().train_mask].to(self.device), 
                    self.data_loader.get_network_torch().edge_index.to(self.device)
                ).to(self.device)

            # Node Classification
            out_nc = self.model.node_classification(z)
            loss_nc = self.criterion( 
                out_nc[self.data_loader.get_network_torch().train_mask].to(self.device),
                self.data_loader.get_network_torch().y[self.data_loader.get_network_torch().train_mask].to(self.device, dtype=torch.int64)
            )
            
            # Link Prediction
            neg_edge_index = negative_sampling(
                edge_index=train.edge_index, 
                num_nodes=train.num_nodes,
                num_neg_samples=train.edge_label_index.size(1), method='sparse'
            )

            edge_label_index = torch.cat(
                [train.edge_label_index, neg_edge_index],
                dim=-1,
            ).to(self.device)
            
            edge_label = torch.cat([
                train.edge_label,
                train.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0).to(self.device)
            
            out_lp = self.model.link_prediction(z, edge_label_index).view(-1)
            loss_lp = self.criterion_lp(
                out_lp.to(self.device), 
                edge_label.to(self.device)
            )
            
            running_loss_nc = loss_nc.item()
            running_loss_lp = loss_lp.item()
            ap_nc_train = average_precision_score(
                    train.y[train.train_mask].cpu().detach().numpy(),
                    out_nc[train.train_mask].cpu().detach().numpy()[:, 1]
                )
            
            ap_lp_train = average_precision_score(
                    edge_label.cpu().detach().numpy(),
                    out_lp.cpu().detach().numpy()
                )

            
            # Phase: EVAL
            self.model.eval()
            z = self.model.encode(
                    self.data_loader.get_network_torch().x[self.data_loader.get_network_torch().val_mask].to(self.device), 
                    self.data_loader.get_network_torch().edge_index.to(self.device)
                )
            
            out_nc = self.model.node_classification(z)
            
            neg_edge_index = negative_sampling(
                    edge_index=val.edge_index, num_nodes=val.num_nodes,
                    num_neg_samples=val.edge_label_index.size(1), method='sparse')

            edge_label_index = torch.cat(
                [val.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                val.edge_label,
                val.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
                
            out_lp = self.model.link_prediction(z, edge_label_index)
            
            ap_nc_val = average_precision_score(
                    val.y[val.val_mask].cpu().detach().numpy(), 
                    out_nc[val.val_mask].cpu().detach().numpy()[:, 1]
            )

            ap_lp_val = average_precision_score(
                    edge_label.to(self.device).cpu().detach().numpy(),
                    out_lp.to(self.device).cpu().detach().numpy()
            )
            
            val_loss_nc = self.criterion(
                out_nc[self.data_loader.get_network_torch().val_mask].to(self.device),
                self.data_loader.get_network_torch().y[self.data_loader.get_network_torch().val_mask].to(self.device, dtype=torch.int64)
            )
            
            val_loss_lp = self.criterion_lp(
                out_lp.to(self.device), 
                edge_label.to(self.device)
            )
            
            
            # Phase: TEST
            z = self.model.encode(
                    self.data_loader.get_network_torch().x[self.data_loader.get_network_torch().test_mask].to(self.device), 
                    self.data_loader.get_network_torch().edge_index.to(self.device)
                )
            
            out_nc = self.model.node_classification(z)
            
            neg_edge_index = negative_sampling(
                    edge_index=test.edge_index, num_nodes=test.num_nodes,
                    num_neg_samples=test.edge_label_index.size(1), method='sparse')

            edge_label_index = torch.cat(
                [test.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                test.edge_label,
                test.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
                
            out_lp = self.model.link_prediction(z, edge_label_index)
            
            
            ap_nc_test = average_precision_score(
                    test.y[test.test_mask].cpu().detach().numpy(), 
                    out_nc[test.test_mask].cpu().detach().numpy()[:, 1]
                )
            
            
            ap_lp_test = average_precision_score(
                    edge_label.to(self.device).cpu().detach().numpy(),
                    out_lp.to(self.device).cpu().detach().numpy()
                )
            
            test_loss_nc = self.criterion(
                out_nc[self.data_loader.get_network_torch().test_mask].to(self.device),
                self.data_loader.get_network_torch().y[self.data_loader.get_network_torch().test_mask].to(self.device, dtype=torch.int64)
            )
            
            test_loss_lp = self.criterion_lp(
                out_lp.to(self.device), 
                edge_label.to(self.device)
            )
            
            # Compute and update AdaDW loss
            loss_trains = [loss_nc, loss_lp]
            loss_vals = [val_loss_nc, val_loss_lp]
            
            loss_tasks = self.adadw(loss_trains, loss_vals)
            loss_tasks.backward()
            
            self.optimizer.step()
            
            # Save accuracy and loss to Tensorboard
            self.writer.add_scalars(
                main_tag='Focal Loss - Node Classification', 
                tag_scalar_dict={
                    'Train': running_loss_nc,
                    'Validation': val_loss_nc,
                    'Test': test_loss_nc
                }, 
                global_step=epoch
            )
            
            self.writer.add_scalars(
                main_tag='Focal Loss - Link Prediction', 
                tag_scalar_dict={
                    'Train': running_loss_lp,
                    'Validation': val_loss_lp,
                    'Test': test_loss_lp
                }, 
                global_step=epoch
            )
            
            self.writer.add_scalars(
                main_tag='AUC-AP - Node Classification',
                tag_scalar_dict={
                    'Train': ap_nc_train,
                    'Validation': ap_nc_val,
                    'Test': ap_nc_test
                    },
                global_step=epoch
            )
            
            self.writer.add_scalars(
                main_tag='AUC-AP - Link Prediction',
                tag_scalar_dict={
                    'Train': ap_lp_train,
                    'Validation': ap_lp_val,
                    'Test': ap_lp_test
                    },
                global_step=epoch
            )
            
            # Show the computed metrics
            self.logger.info('-' * 5 + 'NODE CLASSIFICATION METRICS' + '-' * 5)
            self.logger.info(f'Loss: {running_loss_nc}')
            self.logger.info(f'AP train: {ap_nc_train}')
            self.logger.info(f'AP val: {ap_nc_val}')
            self.logger.info(f'AP test: {ap_nc_test}')
            self.logger.info(20 * '-')
            
            self.logger.info('-' * 5 + 'LINK PREDICTION METRICS' + '-' * 5)
            self.logger.info(f'Loss: {running_loss_lp}')
            self.logger.info(f'AP train: {ap_lp_train}')
            self.logger.info(f'AP val: {ap_lp_val}')
            self.logger.info(f'AP test: {ap_lp_test}')
            self.logger.info(20 * '-')
            
                
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
