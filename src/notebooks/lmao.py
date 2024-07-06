import warnings

warnings.filterwarnings('ignore')

import networkx as nx
import networkit as nk
import numpy as np
import polars as pl
import pandas as pd

import torch.nn as nn
import torch

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import igraph as ig
from torch_geometric.loader import NeighborLoader
import tqdm
from multiprocessing import Pool


from sklearn.metrics import average_precision_score

import os
import sys

sys.path.append('../')
sys.path.append('/Users/phamminhlong/Desktop/paper')

ROOT = os.getcwd()
WORK_DIR = os.path.dirname(os.path.dirname(ROOT))


FIRST_FEAT_NAME = {
        'column_1': 'transid',
        'column_2': 'time_steps',
}

df_classes = pl.read_csv(os.path.join(WORK_DIR, 'data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv'), new_columns=['transid', 'class'])
df_edgelist = pl.read_csv(os.path.join(WORK_DIR, 'data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv'), new_columns=['current_transid', 'next_transid'])
df_features = pl.read_csv(os.path.join(WORK_DIR, 'data/elliptic_bitcoin_dataset/elliptic_txs_features.csv'), has_header=False)

from src.service.data_loader import EllipticLoader
from src.service.graph_model.gat import GAT

e = EllipticLoader(
    path_classes='data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv',
    path_edgelist='data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv',
    path_features='data/elliptic_bitcoin_dataset/elliptic_txs_features.csv'   
)

hidden_dim=64
embedding_dim=128
n_layers=3
n_features=166
output_dim=2
dropout_rate=0.5
heads=5
batch_size=128
lr=1e-4
epochs=500

gat = GAT(
    num_features=n_features,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
    output_dim=output_dim,
    n_layers=n_layers,
    heads=heads,
    dropout_rate=dropout_rate
)

data = e.load()

print(data.get_network_torch())