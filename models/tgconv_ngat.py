import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GATConv, GlobalAttention
from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.nn.norm.graph_norm import GraphNorm

from semi_supervised_AD.models.tgconv_ngat_base import TGConvNGATBase
from semi_supervised_AD.models.temporal_gated_conv import TemporalGatedConv
from semi_supervised_AD.models.temporal_gated_conv import TemporalGatedConvTrans

    
class TGConvNGAT(nn.Module):
    
    def __init__(
        self, 
        num_nodes: int, 
        in_dim: int, 
        conv_hidden_channels,
        conv_out_channels,
        graph_h_dim, 
        out_dim, 
        **kwargs,
    ):
        
        super(TGConvNGAT, self).__init__()
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

#         self.decoder = TemporalGatedConvTrans(in_channels=conv_out_channels,
#                                               out_channels=self.feat_size, 
#                                               kernel_size=self.kernel_size_trans)
        
        self.gn = GraphNorm(in_channels=self.num_seq*conv_out_channels)
        
        self.global_attn =GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(self.num_seq*conv_out_channels, graph_h_dim), 
            nn.ReLU(), 
            nn.Linear(graph_h_dim, 1)))

        self.lin = nn.Linear(self.num_seq*conv_out_channels, graph_h_dim)
        self.graph_aux_lin = nn.Linear(graph_h_dim + self.graph_aux_dim, graph_h_dim)
        self.predict = nn.Sequential(nn.Linear(graph_h_dim, graph_h_dim // 4),
                                     nn.ReLU(),
                                     nn.Linear(graph_h_dim // 4, out_dim))
        
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
    
#     def reconstruct_node(self, node_emb):    
#         batch_size = node_emb.size(0)
#         node_rec = self.decoder(node_emb)
#         node_rec = node_rec.permute(0, 2, 1, 3)
#         node_rec = node_rec.view(batch_size * self.num_nodes, self.seq_len * self.feat_size)
#         return node_rec  # node_rec: [batch_num_nodes, flatten_feat_size]
    
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
#         node_rec = self.reconstruct_node(node_emb)
        out = F.dropout(graph_emb, self.dropout, training=self.training)

        out = self.predict(out)
        
        return node_emb, graph_emb, out
    