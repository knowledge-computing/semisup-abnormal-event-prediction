import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.norm.graph_norm import GraphNorm

from semi_supervised_AD.models.temporal_gated_conv import TemporalGatedConv


class TGConvNGATBase(nn.Module):
    
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.,
        bias: bool = True,
    ):
    
        super(TGConvNGATBase, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dropout = dropout

        self._temporal_conv1 = TemporalGatedConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            num_nodes=num_nodes,
        )

        self._graph_conv = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=self.dropout, 
            bias=True,
        )

        self._temporal_conv2 = TemporalGatedConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_nodes=num_nodes,
        )

        self._graph_norm = GraphNorm(in_channels=hidden_channels)

    def forward(self, x, batch, edge_index, edge_weight=None):

        """
            x: [batch_size, seq_len, num_nodes, in_channels]
            
        returns:
            out: [batch_size, seq_len - kernel_size + 1, num_nodes, in_channels]
        """
        
        # t_0: [batch_size, seq_len - kernel_size, num_nodes, hidden_channels]
        out_0 = self._temporal_conv1(x)  
        
        batch_size, num_seq, num_nodes, hidden_channels = out_0.size()
        out = torch.zeros_like(out_0).to(out_0.device)        
        
        for i in range(num_seq):
            # out_0_i: [batch_size x num_nodes, h_dim]
            out_0_i = out_0[:, i, ...].view(batch_size * num_nodes, hidden_channels)
            
            out_0_i = self._graph_conv(out_0_i, edge_index, edge_weight)
            
            # scatter mean: taking the mean based on the given index
            # returns the mean for each graph and each hidden dim: [num_batch, out_channels]
            # mean = scatter_mean(T_0_t, batch, dim=0, dim_size=batch_size)[batch]
            out_0_i = self._graph_norm(out_0_i, batch=batch)
            out[:, i, ...] = out_0_i.view(batch_size, num_nodes, -1)

        out = F.relu(out)
        out = self._temporal_conv2(out)
        return out