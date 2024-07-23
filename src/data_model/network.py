import torch
from torch_geometric.data import Data
from typing import Dict, List
import networkx as nx
import networkit as nk
import numpy as np
import polars as pl
import pandas as pd


from src.data_model.graph import Graph


class DataNetWork:


    def __init__(self,
        df_features: pl.DataFrame, 
        df_edges: pl.DataFrame,  
        train_mask: np.array, 
        val_mask: np.array, 
        test_mask: np.array, 
        directed: bool = False):

        self.df_features = df_features
        self.df_edges = df_edges
        self.directed = directed
        
        self.graph: Graph = self._set_up_network_info()

        self.fraud_dict = dict(
            zip(
                pl.from_pandas(df_features["transid"].to_pandas().map(self.graph.map_id)),
                df_features["class"]
                )
            ) # gan class voi id tuong ung
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        
        
    def _set_up_network_info(self) -> Graph:
        nodes = self.df_features.select(pl.col("transid"))
        
        
        map_id = {i:j for i,j in enumerate((nodes
                                            .to_series()
                                            .to_list()))} # Gan tung transid voi index tuong ung
        edges = self.df_edges.select(
            pl.col('current_transid'),
            pl.col('next_transid')
        )
        
        if not self.directed:
            map_id = {j:i for i,j in enumerate((nodes
                                            .to_series()
                                            .to_list()))} 

            
            nodes = nodes.to_pandas()
            nodes['transid'] = nodes['transid'].map(map_id).astype(np.int64)
            nodes = pl.from_pandas(nodes)
            
            
            edges = edges.to_pandas()
            
            edges_direct = edges[['current_transid', 'next_transid']]
            edges_reverse = edges_direct[['next_transid', 'current_transid']]
            edges_reverse.columns = ['current_transid', 'next_transid']
            
            edges = pd.concat([edges_direct, edges_reverse], axis=0) # ket hop 2 cai lai de undirect ???
            
            edges['current_transid'] = edges['current_transid'].map(map_id).astype(np.int64)
            
            edges['next_transid'] = edges['next_transid'].map(map_id).astype(np.int64)
            edges = pl.from_pandas(edges)
            

        
        return Graph(
            nodes=nodes,
            edges=edges,
            map_id=map_id
        )
        
    def get_network_torch(self) -> Data:
        labels = self.df_features['class']
        features = self.df_features.to_pandas().drop(columns=['transid', 'class'])
        
        x = torch.tensor(np.array(features.to_numpy(), dtype=float), dtype=torch.float)
        if x.size()[1] == 0:
            x = torch.ones(x.size()[0], 1)
        
        x = x[:, 1:94]
        y = torch.tensor(np.array(labels.to_numpy(), dtype=np.int64), dtype=torch.int64)
        
        # Reformat and convert to tensor
        edge_index = np.array(self.graph.edges.to_numpy()).T 
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        #create weights tensor with same shape of edge_index
        weights = torch.tensor([1]* edge_index.shape[1] , dtype=torch.float) 
        
        # Create pyG dataset
        data = Data(x=x, y=y, edge_index=edge_index)

        if self.train_mask is not None:
            data.train_mask = torch.tensor(self.train_mask, dtype=torch.bool)
        if self.val_mask is not None:
            data.val_mask = torch.tensor(self.val_mask, dtype=torch.bool)
        if self.test_mask is not None:
            data.test_mask = torch.tensor(self.test_mask, dtype=torch.bool)
        
        return data 