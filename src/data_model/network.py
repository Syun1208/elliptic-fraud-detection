import torch
from torch_geometric.data import Data
from typing import Dict, List
import networkx as nx
import networkit as nk
import numpy as np
import polars as pl


from src.data_model.graph import Graph



class DataNetWork:
    
    def __init__(
        self, 
        df_features: pl.DataFrame, 
        df_edges: pl.DataFrame,  
        train_mask: np.array, 
        val_mask: np.array, 
        test_mask: np.array, 
        directed: bool = False
    ):
        
        self.df_features = df_features
        self.df_edges = df_edges
        self.directed = directed
        
        self.graph: Graph = self._set_up_network_info()

        self.fraud_dict = dict(
            zip(
                df_features["transid"].map_dict(self.graph.map_id),
                df_features["class"]
                )
            )
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        
        
        
    def _set_up_network_info(self) -> Graph:
        nodes = self.df_features.select(
            pl.col('transid')
        )
        
        map_id = {i:j for i,j in enumerate((nodes
                                            .to_series()
                                            .to_list()))} 
        
        edges = self.df_edges.select(
            pl.col('current_transid'),
            pl.col('next_transid')
        )
        if not self.directed:
            map_id = {j:i for i,j in enumerate((nodes
                                            .to_series()
                                            .to_list()))} 
            
            nodes = nodes.with_columns(
                pl.col('transid').map_dict(map_id).cast(pl.Int64)
            )
            edges = edges.with_columns(
                pl.col('current_transid').map_dict(map_id).cast(pl.Int64),
                pl.col('next_transid').map_dict(map_id).cast(pl.Int64)
            )
        
        return Graph(
            nodes=nodes,
            edges=edges,
            map_id=map_id
        )
        
        
        
    def get_network_nx(self) -> nx.DiGraph:
        edges_zipped = zip(self.graph.edges['current_transid'], self.graph.edges['next_transid'])
        
        if self.directed:
            G_nx = nx.DiGraph()
        else: 
            G_nx = nx.Graph()
        
        G_nx.add_nodes_from(self.graph.nodes)
        G_nx.add_edges_from(edges_zipped)
        
        return G_nx     
            
            
            
    def get_network_nk(self) -> nx.DiGraph:
        edges_zipped = zip(self.graph.edges['current_transid'], self.graph.edges['next_transid'])
        
        G_nk = nk.Graph(len(self.graph.nodes), directed = self.directed)
        
        for u,v in edges_zipped:
            G_nk.addEdge(u,v)
            
        return G_nk 
        
        
        
    def get_network_torch(self) -> Data:
        labels = self.df_features['class']
        features = self.df_features.drop(columns=['transid', 'class'])
        
        x = torch.tensor(np.array(features.to_numpy(), dtype=float), dtype=torch.float)
        if x.size()[1] == 0:
            x = torch.ones(x.size()[0], 1)
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
    
    
    
    def get_features(
            self, 
            full=False
        ) -> pl.DataFrame:
        
        if full:
            X = self.df_features[self.df_features.columns[2: 167]]
        else:
            X = self.df_features[self.df_features.columns[2: 95]]
            
        return X
    
    
    
    def get_features_torch(
        self, 
        full=False
    ) -> torch.tensor:
        
        X = self.get_features(full)
        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        
        return(X)



    def get_train_test_split_intrinsic(
        self, 
        train_mask: np.array, 
        test_mask: np.array, 
        device: str = 'cpu'
    ) -> List[torch.tensor]:
        
        X: pl.DataFrame = self.get_features()
        y: pl.Series = self.df_features['class']

        X_train = X.filter(
            pl.Series(train_mask.tolist())
        )
        y_train = y.filter(
            pl.Series(train_mask.tolist())
        )

        X_test = X.filter(
            pl.Series(test_mask.tolist())
        )
        y_test = y.filter(
            pl.Series(test_mask.tolist())
        )

        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long).to(device)

        X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long).to(device)

        return X_train, y_train, X_test, y_test



    def get_fraud_dict(self) -> Dict[int, int]:
        return self.fraud_dict
    
    
    
    def get_masks(self) -> List[np.array]:
        return self.train_mask, self.val_mask, self.test_mask