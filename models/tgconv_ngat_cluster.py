import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import Parameter

from torch_geometric.nn import global_add_pool, GATConv, GlobalAttention
from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.nn.norm.graph_norm import GraphNorm

import sys
sys.path.append('../../')

from models.tgconv_ngat_base import TGConvNGATBase
from models.temporal_gated_conv import TemporalGatedConv
from models.temporal_gated_conv import TemporalGatedConvTrans


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.
        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.
        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
    
    
class TGConvNGATCluster(nn.Module):
    
    def __init__(
        self, 
        num_nodes: int, 
        in_dim: int, 
        conv_hidden_channels,
        conv_out_channels,
        graph_h_dim, 
        out_dim, 
        cluster_number,
        **kwargs,
    ):
        
        super(TGConvNGATCluster, self).__init__()
        self.graph_aux_dim = kwargs.get('graph_aux_dim', 0)
        self.aggr_func = kwargs.get('aggr_func', 'add_pool')
        self.dropout = kwargs.get('dropout', 0.)

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.conv_hidden_channels = conv_hidden_channels
        self.conv_out_channels = conv_out_channels
        self.graph_h_dim = graph_h_dim
        self.out_dim = out_dim
        self.feat_size = 4
        self.seq_len = 72
        self.kernel_size = 12
        self.kernel_size_trans = 12 * 2 - 1
        self.num_seq = self.seq_len - 2 * (self.kernel_size - 1)
        
        self.encoder = TGConvNGATBase(num_nodes=num_nodes,
                                      in_channels=self.feat_size, 
                                      hidden_channels=conv_hidden_channels,
                                      out_channels=conv_out_channels,
                                      kernel_size=self.kernel_size)

        self.decoder = TemporalGatedConvTrans(in_channels=conv_out_channels,
                                              out_channels=self.feat_size, 
                                              kernel_size=self.kernel_size_trans)
        
        self.gn = GraphNorm(in_channels=self.num_seq*conv_out_channels)
        
        self.global_attn =GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(self.num_seq*conv_out_channels, graph_h_dim), 
            nn.ReLU(), 
            nn.Linear(graph_h_dim, 1)))

        self.lin = nn.Linear(self.num_seq*conv_out_channels, graph_h_dim)
        self.graph_aux_lin = nn.Linear(graph_h_dim + self.graph_aux_dim, graph_h_dim)
        self.predict = nn.Linear(graph_h_dim, out_dim)
        
#         self.cluster_number = cluster_number
#         self.alpha = 1
#         self.assignment = ClusterAssignment(
#             cluster_number, graph_h_dim, self.alpha
#         )
        
    def embed_node(self, input_data):
        x = input_data.x[:, :self.seq_len * self.feat_size]  # x: [batch_num_nodes, flatten_feat_size]
        x = x.view(-1, self.num_nodes, self.seq_len, self.feat_size)       
        x = x.permute(0, 2, 1, 3)  # x: [batch_size, seq_len, num_nodes, feat_size]
        
        batch_size = x.size(0)
        
        # node_emb: [batch_size, num_seq, num_nodes, conv_out_channels]
        node_emb = self.encoder(x, batch=input_data.batch, 
                                edge_index=input_data.edge_index)
        
        node_emb = node_emb.permute(0, 2, 1, 3)  # node_emb: [batch_size, num_nodes, num_seq, conv_out_channels]
        node_emb = node_emb.view(batch_size * self.num_nodes, self.num_seq * self.conv_out_channels)
        node_emb = self.gn(node_emb, batch=input_data.batch)
        node_emb = node_emb.view(batch_size, self.num_nodes, self.num_seq, self.conv_out_channels)
        node_emb = node_emb.permute(0, 2, 1, 3)
        return node_emb  # node_emb: [batch_size, num_seq, num_nodes, conv_out_channels]
    
    def reconstruct_node(self, node_emb):  
        batch_size = node_emb.size(0)
        node_rec = self.decoder(node_emb)
        node_rec = node_rec.permute(0, 2, 1, 3)
        node_rec = node_rec.view(batch_size * self.num_nodes, self.seq_len * self.feat_size)
        return node_rec  # node_rec: [batch_num_nodes, flatten_feat_size]
    
    def embed_graph(self, node_emb, batch):
    
        batch_size = node_emb.size(0)
        node_emb = node_emb.permute(0, 2, 1, 3)  # node_emb: [batch_size, num_nodes, num_seq, conv_out_channels]
        node_emb = node_emb.view(batch_size * self.num_nodes, self.num_seq * self.conv_out_channels)
    
        if self.aggr_func == 'add_pool':
            graph_emb = global_add_pool(node_emb, batch)
        else:
            graph_emb = self.global_attn(node_emb, batch=batch)     
        
        graph_emb = self.lin(F.relu(graph_emb))
        
        if self.graph_aux_dim:
            graph_emb = torch.cat([graph_emb, input_data.graph_attributes], dim=-1)     
            graph_emb = self.graph_aux_lin(F.relu(graph_emb))

        return graph_emb  # graph_emb: [batch_size x graph_h_dim]
    
    def forward(self, input_data):
        
        node_emb = self.embed_node(input_data)
        graph_emb = self.embed_graph(node_emb, input_data.batch)
        node_rec = self.reconstruct_node(node_emb)
        
        out = F.dropout(graph_emb, self.dropout, training=self.training)
        out = self.predict(out)
        
#         assignment = self.assignment(graph_emb)
        
        return node_emb, node_rec, graph_emb, out
    