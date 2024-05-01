# -*- coding: gbk -*-
#Building the GNN

from torch.nn import Linear, BatchNorm1d
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, ImbalancedSampler
from torch_geometric.nn import GCNConv, GATConv, TopKPooling, BatchNorm, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.optim as optim
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import datasets
from torch.utils.data import WeightedRandomSampler

class NEW(InMemoryDataset):

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
            print(node_id)

            attributes = node_attrs.loc[node_id + 1, :]

            edges = edge_index.loc[edge_index['source_node'].isin(node_id)]
            edges_ids = edges.index

            label = graphID.loc[graph_no, 'label']
            print(edges)



            edges_ids = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            map_dict = {v.item():i for i,v in enumerate(torch.unique(edges_ids))}
            map_edge = torch.zeros_like(edges_ids)
            for k,v in map_dict.items():
                map_edge[edges_ids==k] = v
            map_dict, map_edge, map_edge.shape
            edges_ids = map_edge.long()
            print("EI")
            print(edges_ids)
            print("EI")
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



dataset = NEW(root='')
dataset = dataset.shuffle()

labels = dataset.data.y

train_indices = []
test_indices = []



for label in set(labels.numpy()):
    label_indices = (labels == label).nonzero(as_tuple=False).view(-1)
    train_size = int(len(label_indices) * train_ratio)
    test_size = len(label_indices) - train_size
    chosen_indices = np.random.permutation(label_indices)
    train_indices.extend(chosen_indices[:train_size].tolist())
    test_indices.extend(chosen_indices[train_size:].tolist())

train_dataset = dataset.index_select(torch.tensor(train_indices))
test_dataset = dataset.index_select(torch.tensor(test_indices))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

sampler = ImbalancedSampler(train_dataset)


print(f'Number of training classes: {train_dataset.num_classes}')
print(f'Number of test classes : {test_dataset.num_classes}')

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(1500, 128)
        self.pool1 = TopKPooling(128, ratio=0.5)
        self.bn1 = BatchNorm1d(128)

        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.5)
        self.bn2 = BatchNorm1d(128)

        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.5)
        self.bn3 = BatchNorm1d(128)

        self.conv4 = GraphConv(128, 128)
        self.pool4 = TopKPooling(128, ratio=0.5)
        self.bn4 = BatchNorm1d(128)

        self.conv5 = GraphConv(128, 128)
        self.pool5 = TopKPooling(128, ratio=0.5)
        self.bn5 = BatchNorm1d(128)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64 )
        self.lin3 = torch.nn.Linear(64, 4)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1+x2+x3+x4+x5
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x



def initialize_model(model_to_process: GCN, num_classes, num_features):
    set_parameter_requires_grad(model_to_process)
    in_features = model_to_process.lin3.in_features
    out_channels = model_to_process.conv1.out_channels
    model_to_process.conv1 = GraphConv(new_features, out_channels)
    model_to_process.lin3 = torch.nn.Linear(in_features, num_classes)
    model_to_process.pool1 = TopKPooling(128, ratio=0.5)
    model_to_process.bn1 = BatchNorm1d(128)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN()
state_dict = torch.load('', map_location=device)
model.load_state_dict(state_dict)
new_features = dataset.num_features
new_classes = dataset.num_classes
initialize_model(model, new_classes, new_features)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=)

weights = torch.ones(4, dtype=torch.float32).to(device)
weights[0] = 1
weights[3] = 1
weights[1] = 1
weights[2] = 1

