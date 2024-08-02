import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, add_remaining_self_loops, softmax
from torch_scatter import scatter_add
from torch.nn import Linear



class GATTuning(MessagePassing):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        weight, 
        att, 
        bias, 
        heads=1, 
        dropout=0, 
        concat=True,    
        negative_slope=0.2, 
        **kwargs
    ) -> None:
        super(GATTuning, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = int(out_channels / heads)
        self.heads = heads
        self.concat = True
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = weight
        self.att = att
        self.bias = bias

    def forward(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
