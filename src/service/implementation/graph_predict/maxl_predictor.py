import torch
import os
import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.utils import get_device
from src.utils.timer import time_complexity
from src.service.abstraction.graph_predict import Predictor
from src.service.abstraction.data_loader import DataLoader
from src.utils.logger import Logger
from src.data_model.score import Score
from src.utils.utils import resample_testmask


FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]
    
    
    
class MAXLPredictorImpl(Predictor):
    
    def __init__(
        self,
        model: torch.nn,
        data_loader: DataLoader,
        logger: Logger,
        device_id: int,
        batch_size: int,
        path_model: str,
        n_random_samples: int
        
    ) -> None:
        
        self.model = model
        self.path_model = path_model
        self.batch_size = batch_size
        self.n_random_samples = n_random_samples
        
        self.data_loader = data_loader.load()
        
        self.logger = logger.get_tracking(__name__)
        
        self.device = get_device(device_id)
    
    
    @time_complexity(name_process='PHASE TEST')
    def predict(self) -> Score:
        
        ra_list = []
        ap_list = []
        
        self.model.to(self.device)

        self.load_model(
            os.path.join(WORK_DIR, self.path_model)
        )
        

        self.model.eval()
        
        for _ in tqdm.tqdm(range(self.n_random_samples), colour='green', desc='Testing: '):
            
            ra_score = 0.0
            ap_score = 0.0
            
            random_test_mark = resample_testmask(self.data_loader.test_mask)
            data = self.data_loader.get_network_torch()
                
            z = self.model.encode(
                data.x.to(self.device), 
                data.edge_index.to(self.device)
            )
            out = self.model.node_classification(z)
            
            y_hat = out[random_test_mark].to(self.device)
            y = data.y[random_test_mark].to(self.device)
            
            ra_score = roc_auc_score(
                y.cpu().detach().numpy(), 
                y_hat.cpu().detach().numpy()[:, 1]
            )
            ap_score = average_precision_score(
                y.cpu().detach().numpy(), 
                y_hat.cpu().detach().numpy()[:, 1]
            )
            
            ra_list.append(ra_score)
            ap_list.append(ap_score)

            
        return Score(
            ap_scores=ap_list,
            ra_scores=ra_list
        )
        
            
        
    def load_model(self, path) -> None:
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(
            checkpoint['model_state_dict']
        )
        
        self.logger.info(f'LOAD MODEL SUCCESSFULLY AT: {path}')