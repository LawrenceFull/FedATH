from os import path as osp
from typing import Callable, List, Optional
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.io import fs
import pickle
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
from torch_geometric.utils import coalesce, cumsum, remove_self_loops
from typing import Dict, List, Optional, Tuple

class WikiPages(InMemoryDataset):
    url = "https://data.dgl.ai/dataset"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name # [chameleon, squirrel]

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ["out1_graph_edges.txt", "out1_node_feature_label.txt"]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        fs.cp(f"{self.url}/{self.name.lower()}.zip", self.raw_dir, extract=True)

    def process(self) -> None:
        edge_index_path = osp.join(self.raw_dir, "out1_graph_edges.txt")
        data_list = []
        with open(edge_index_path, 'r') as file:
            # Skip the header
            next(file)
            for line in file:
                data_list.append([int(number) for number in line.split()])
        edge_index = torch.tensor(data_list).long().T

        node_feature_label_path = osp.join(self.raw_dir, "out1_node_feature_label.txt")
        node_feature_list = []
        node_label_list = []
        with open(node_feature_label_path, 'r') as file:
            # Skip the header
            next(file)
            for line in file:
                node_id, feature, label = line.strip().split('\t')
                node_feature_list.append([int(num) for num in feature.split(',')])
                node_label_list.append(int(label))
        x = torch.tensor(node_feature_list)
        y = torch.tensor(node_label_list)
        data = Data(x=x, edge_index=edge_index, y=y)
        
        
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'