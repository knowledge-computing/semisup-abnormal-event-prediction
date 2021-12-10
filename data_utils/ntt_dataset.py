import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


def read_ntt_data(data_path):
    
    data = np.load(data_path, allow_pickle=True)
    num_graphs = data['num_graphs']
    node_attr = data['node_attr']
    edge_index = data['edge_index']
    edge_weight = data['edge_weight']
    node_label = data['node_label']
    node_auxiliary_attr = data['node_auxiliary_attr']
    graph_auxiliary_attr = data['graph_auxiliary_attr']
    graph_label = data['graph_label']    
    time_interval = data['time_interval']
    num_nodes = len(data['node_name'])
    
    assert len(edge_index) == len(graph_label) == len(node_attr) - time_interval
        
    data_list = []
    for i in range(num_graphs):
        node_x = np.stack(node_attr[i: (i+time_interval)])
        node_x = np.swapaxes(node_x, 0, 1).reshape(num_nodes, -1)  # flatten

        data = Data(x=torch.tensor(node_x, dtype=torch.float), 
                    node_label=torch.tensor(node_label[i], dtype=torch.float),
                    edge_index=torch.tensor(edge_index[i].T, dtype=torch.long), 
                    edge_weight=torch.tensor(edge_weight[i], dtype=torch.float).view(-1), 
                    y=torch.tensor([graph_label[i]], dtype=torch.long), 
                    node_aux_attr=torch.tensor(node_auxiliary_attr, dtype=torch.float), 
                    graph_aux_attr=torch.tensor(graph_auxiliary_attr[i: (i+time_interval)], dtype=torch.float).view(1, -1), 
                    graph_id=torch.tensor([i], dtype=torch.long))
        data_list.append(data)
    return data_list


class NTTDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NTTDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        return [os.path.join(self.root, 'graph.npz')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_ntt_data(self.raw_file_names[0])
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
