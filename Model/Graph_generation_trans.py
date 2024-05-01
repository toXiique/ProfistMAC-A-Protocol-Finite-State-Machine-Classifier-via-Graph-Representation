# -*- coding: gbk -*-
import torch
import os
import os.path as osp
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
train_path = ""

class Model(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):

        return ['edge.csv', 'graphid2label.csv', 'node2graphID.csv', 'nodeattrs.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        path = os.path.join(self.raw_dir, 'nodeattrs.csv')
        node_attrs = pd.read_csv(path, sep=',', header=0, index_col=0)
        node_attrs.index += 1

        path = os.path.join(self.raw_dir, 'edge.csv')
        edge_index = pd.read_csv(path, sep=',', header=0)
        edge_index.index += 1

        path = os.path.join(self.raw_dir, 'node2graphID.csv')
        nodes = pd.read_csv(path, sep=',', header=0)
        nodes.index += 1

        path = os.path.join(self.raw_dir, 'graphid2label.csv')
        graphID = pd.read_csv(path, sep=',', header=0)
        graphID.index += 1

        data_list = []

        ids_list = nodes['graph_id'].unique()

        for graph_no in tqdm(ids_list):

            node_id = nodes.loc[nodes['graph_id']==graph_no, 'node_id']

            attributes = node_attrs.loc[node_id + 1, :]

            edges = edge_index.loc[edge_index['source_node'].isin(node_id)]
            edges_ids = edges.index

            label = graphID.loc[graph_no, 'label']

            edges_ids = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            map_dict = {v.item():i for i,v in enumerate(torch.unique(edges_ids))}
            map_edge = torch.zeros_like(edges_ids)
            for k,v in map_dict.items():
                map_edge[edges_ids==k] = v
            map_dict, map_edge, map_edge.shape
            edges_ids = map_edge.long()
            label = torch.tensor(label, dtype=torch.long)

            attrs = torch.tensor(attributes.to_numpy(),dtype=torch.float)

            graph = Data(x=attrs, edge_index=edges_ids, y=label)

            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



import torch_geometric.transforms as T

dataset = NEW(root='')
dataset_test = NEW(root='')

