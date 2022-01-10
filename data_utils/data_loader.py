import os
import logging
import numpy as np
import torch
from torch_geometric.data import DataLoader, Dataset

import sys
sys.path.append('../../')

from data_utils.ntt_dataset import NTTDataset
from data_utils.gen_train_val_test import split_train_val_test
from data_utils.gen_train_val_test import compute_label_weight


class ConcatDataset(Dataset):
    
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)


class NTTDataLoader():
    
    def __init__(self, dataset: NTTDataset, 
                 num_classes: int = 2,
                 batch_size: int = 16,
                 use_unlabel: bool = False,
                 shuffle: bool = True,
                 random_seed: int = 1234):
        
        self.dataset = dataset
        self.train_dataset, self.val_dataset, self.test_dataset, self.unsup_dataset = \
            split_train_val_test(dataset, 
                                 num_classes=num_classes, 
                                 shuffle=shuffle, 
                                 use_unlabel=use_unlabel, 
                                 random_seed=random_seed)
        
        if use_unlabel:
            self.semi_loader = DataLoader(ConcatDataset(self.train_dataset, self.unsup_dataset), 
                                          batch_size=batch_size, shuffle=True)
        else:
            self.semi_loader = DataLoader(self.train_dataset, 
                                          batch_size=batch_size, shuffle=True)

        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, 
                                     batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, 
                                      batch_size=batch_size, shuffle=False)
        self.unsup_loader = DataLoader(self.unsup_dataset, 
                                       batch_size=batch_size, shuffle=False)
        
        self.num_nodes = dataset[0].num_nodes
        self.num_features = dataset.num_features  
        self.node_aux_dim = dataset[0].node_aux_attr.size(-1)
        self.graph_aux_dim = dataset[0].graph_aux_attr.size(-1)

        ''' compute the weight for labels '''
        label_weight = compute_label_weight(self.train_dataset, num_classes)
        self.label_weight = torch.tensor(label_weight, dtype=torch.float)
        
