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
    
    def __init__(
        self, 
        df_features: pl.DataFrame, 
        df_edges: pl.DataFrame,  
        df_classes: pl.DataFrame,
        train_mask: np.array, 
        val_mask: np.array, 
        test_mask: np.array, 
        directed: bool = False
    ):
        
        self.df_features = df_features
        self.df_edges = df_edges
        self.df_classes = df_classes
        
        self.directed = directed
        
        self.graph: Graph = self._set_up_network_info()

        self.fraud_dict = dict(
            zip(
                pl.from_pandas(df_features["transid"].to_pandas().map(self.graph.map_id)),
                df_features["class"]
                )
            )
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        

    def __process_edge4link_prediction(self, df_edges: pl.DataFrame) -> List[torch.Tensor]:
        
        df_edges = df_edges.rename({'current_transid': 'transid'})
        df_edges = df_edges.join(self.df_classes, on='transid', how='left')
        df_edges = df_edges.rename({'transid': 'current_transid', 'class': 'current_class', 'next_transid': 'transid'})
        df_edges = df_edges.join(self.df_classes, on='transid', how='left')
        df_edges = df_edges.rename({'transid': 'next_transid', 'class': 'next_class'})

        edge_info = (df_edges.with_columns(
                pl.when(pl.col('current_class') == pl.col('next_class'))
                .then(1)
                .otherwise(0)
                .alias('edge_label')
            )
            .filter(
                pl.col('edge_label') == 1
            )
            .drop(
                ['current_class', 'next_class']
            )
        )
        
        edge_label_index = torch.from_numpy(
            edge_info.select(['current_transid', 'next_transid']).to_numpy()
        ).t()
        
        edge_label = torch.from_numpy(
            edge_info.select(['edge_label']).to_numpy()
        ).t()
        
        return edge_label, edge_label_index
      
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
        edge_label, edge_label_index = self.__process_edge4link_prediction(df_edges=edges)
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
            
            edges = pd.concat([edges_direct, edges_reverse], axis=0)
            
            edge_label, edge_label_index = self.__process_edge4link_prediction(df_edges=edges)
            
            edges['current_transid'] = edges['current_transid'].map(map_id).astype(np.int64)
            edges['next_transid'] = edges['next_transid'].map(map_id).astype(np.int64)
            edges = pl.from_pandas(edges)
            

        
        return Graph(
            nodes=nodes,
            edges=edges,
            map_id=map_id,
            edge_label=edge_label,
            edge_label_index=edge_label_index
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
        data = Data(
            x=x, 
            y=y, 
            edge_index=edge_index, 
            edge_label_index=self.graph.edge_label_index, 
            edge_label=self.graph.edge_label
        )

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