from pathlib import Path
import polars as pl
import torch
import os
from abc import ABC, abstractmethod

from src.utils.timer import time_complexity
from src.utils.constants import FIRST_FEAT_NAME
from src.data_model.network import DataNetWork


FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]


class DataLoader(ABC):
    
    @abstractmethod
    def load(self):
        pass


class EllipticLoader(DataLoader):
    
    def __init__(
        self,
        path_features: str,
        path_edgelist: str,
        path_classes: str
    ) -> None:
        
        self.path_features = path_features
        self.path_edgelist = path_edgelist
        self.path_classes = path_classes
    
    
    @time_complexity(name_process='PHASE ELLIPTIC LOADER')
    def load(self) -> DataNetWork:
        
        feat_df = pl.read_csv(
            os.path.join(WORK_DIR, self.path_features), 
            has_header=False
        )
    
        second_feat_name = {f'column_{i}': f'feature_{i-2}' for i in range(3, feat_df.shape[1] + 1)}
        converted_feature_names = {**FIRST_FEAT_NAME, **second_feat_name}
        feat_df = feat_df.rename(converted_feature_names)

        edge_df = pl.read_csv(
            os.path.join(WORK_DIR, self.path_edgelist), 
            new_columns=['current_transid', 'next_transid']
        
        )
        class_df = pl.read_csv(
            os.path.join(WORK_DIR, self.path_classes),
            new_columns=['transid', 'class']
        )

        mapping = {'unknown': 2, '1': 1, '2': 0}
        mapper = pl.DataFrame({
            "class": list(mapping.keys()),
            "new_class": list(mapping.values())
        })
        class_df = class_df.join(mapper, on='class', how='left').drop('class').rename({'new_class': 'class'})
        y = torch.from_numpy(class_df['class'].to_numpy())

        # Timestamp based split:
        time_step = torch.from_numpy(feat_df['time_steps'].to_numpy())
        train_mask = (time_step < 30) & (y != 2)
        val_mask = (time_step >= 30) & (time_step < 40) & (y != 2) 
        test_mask = (time_step >= 40) & (y != 2)

        network = DataNetWork(
            feat_df, 
            edge_df, 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask
        )

        return network
