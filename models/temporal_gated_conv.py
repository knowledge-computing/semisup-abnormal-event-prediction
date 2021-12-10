import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGatedConv(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_nodes: int,
    ):
        """
            in_channels (int): number of input features
            out_channels (int): number of output features
            kernel_size (int): convolutional kernel size
        """
        
        super(TemporalGatedConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self._batch_norm = nn.BatchNorm2d(num_nodes)
        
    def forward(self, x):
        """
            x: [batch_size, num_seq, num_nodes, in_channels]
            
        returns:
            h: [batch_size, num_seq - kernel_size + 1, num_nodes, out_channels]
        """
        x = x.permute(0, 3, 2, 1)
        p = self.conv_1(x)
        q = torch.sigmoid(self.conv_2(x))
        pq = p * q
        h = F.relu(pq + self.conv_3(x))
        h = h.permute(0, 3, 2, 1)  # h: [batch_size, seq_len - kernel_size + 1, num_nodes, out_channels]
        
        # normalization
        h = h.permute(0, 2, 1, 3)
        h = self._batch_norm(h)
        h = h.permute(0, 2, 1, 3)
        return h
    
    
class TemporalGatedConvTrans(nn.Module):
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        """
            in_channels (int): number of input features
            out_channels (int): number of output features
            kernel_size (int): convolutional kernel size
        """

        super(TemporalGatedConvTrans, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size))
    
    def forward(self, x):
        """
            x: [batch_size, seq_len, num_nodes, in_channels]
            
        returns:
            h: [batch_size, seq_len + kernel_size - 1, num_nodes, out_channels]
        """
        x = x.permute(0, 3, 2, 1)
        p = self.conv_1(x)
        q = torch.sigmoid(self.conv_2(x))
        pq = p * q
        h = F.relu(pq + self.conv_3(x))
        h = h.permute(0, 3, 2, 1)
        return h