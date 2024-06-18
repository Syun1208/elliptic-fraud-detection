from pathlib import Path
import polars as pl
import torch
import os

from src.utils.constants import FIRST_FEAT_NAME
from service.data_processing import DataNetWork


FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]

def load_elliptic() -> DataNetWork:
    
    feat_df = pl.read_csv(os.path.join(WORK_DIR, 'data/elliptic_bitcoin_dataset/elliptic_txs_features.csv'), has_header=False)
    
    second_feat_name = {f'column_{i}': f'feature_{i-2}' for i in range(3, feat_df.shape[1] + 1)}
    converted_feature_names = {**FIRST_FEAT_NAME, **second_feat_name}
    feat_df = feat_df.rename(converted_feature_names)

    edge_df = pl.read_csv(os.path.join(WORK_DIR, 'data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv'), new_columns=['current_transid', 'next_transid'])
    class_df = pl.read_csv(os.path.join(WORK_DIR, 'data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv'), new_columns=['transid', 'class'])

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
    
    ntw = DataNetWork(feat_df, edge_df, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return ntw
