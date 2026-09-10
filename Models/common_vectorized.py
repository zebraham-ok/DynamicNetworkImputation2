"""
Shared vectorized components:
    - precompute_adj_matrices: precompute the normalized sparse adjacency matrices for all time steps
    - GCNEncoderVectorized: vectorized GCN encoder, uses torch.sparse.mm instead of GCNConv
    - GATEncoderVectorized: vectorized GAT encoder, precomputes edge indices to avoid rebuilding Data objects
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv
from Models.common import FCEmbedding


def precompute_adj_matrices(dynamic_data, time_steps, num_nodes, device):
    """Precompute the normalized sparse adjacency matrices A_norm = D^{-1/2} A D^{-1/2} for all time steps; returns List[sparse_coo_tensor]."""
    adj_matrices = []
    for t in time_steps:
        mask = dynamic_data.edge_time == t
        edge_index = dynamic_data.edge_index[:, mask]

        if edge_index.size(1) > 0:
            row, col = edge_index
            deg = torch.bincount(row, minlength=num_nodes).float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            adj = torch.sparse_coo_tensor(
                edge_index, norm,
                size=(num_nodes, num_nodes)
            )
        else:
            # Empty time step -> empty sparse tensor
            adj = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,)),
                size=(num_nodes, num_nodes)
            )

        adj_matrices.append(adj.coalesce().to(device))

    return adj_matrices


class GCNEncoderVectorized(nn.Module):
    """Vectorized GCN encoder: precomputed adjacency matrices + torch.sparse.mm instead of GCNConv."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3,
                 use_fc_embedding=False, fc_embed_dim=128):
        super().__init__()
        self.use_fc_embedding = use_fc_embedding
        self.num_layers = num_layers
        self.dropout = dropout

        if use_fc_embedding:
            self.fc_embed = FCEmbedding(input_dim, fc_embed_dim, dropout)
            _input_dim = fc_embed_dim
        else:
            self.fc_embed = None
            _input_dim = input_dim

        # Store the weights and biases of each layer, replacing GCNConv
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        layer_dims = [_input_dim] + [hidden_dim] * max(0, num_layers - 1) + [output_dim]
        if num_layers == 1:
            layer_dims = [_input_dim, output_dim]
        else:
            layer_dims = [_input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            self.weights.append(nn.Parameter(torch.Tensor(layer_dims[i], layer_dims[i+1])))
            self.biases.append(nn.Parameter(torch.zeros(layer_dims[i+1])))
            self._reset_param(self.weights[-1])

    def _reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, adj_matrices, node_feats):
        """adj_matrices: List[T] adjacency matrices [N, N]; node_feats: [N, input_dim], shared across all time steps. Returns [T, N, output_dim]."""
        T = len(adj_matrices)

        # FC dimension reduction is done only once, then broadcast to all time steps
        if self.fc_embed is not None:
            x = self.fc_embed(node_feats)  # [N, fc_dim]
        else:
            x = node_feats  # [N, input_dim]

        for layer_idx in range(self.num_layers):
            W, b = self.weights[layer_idx], self.biases[layer_idx]
            is_last = (layer_idx == self.num_layers - 1)

            if layer_idx == 0:
                # First layer: x is [N, F], shared across all time steps -> do the linear transform only once
                x_proj = x @ W  # [N, H_out]
                out = []
                for t in range(T):
                    if adj_matrices[t]._nnz() > 0:
                        agg = torch.sparse.mm(adj_matrices[t], x_proj)
                    else:
                        agg = torch.zeros_like(x_proj)
                    out.append(agg.unsqueeze(0))
                x = torch.cat(out, dim=0) + b  # [T, N, H_out]
            else:
                # Subsequent layers: x is [T, N, H_in], different per time step
                x_proj = x @ W  # [T, N, H_out], a single matmul handles all time steps
                out = []
                for t in range(T):
                    if adj_matrices[t]._nnz() > 0:
                        agg = torch.sparse.mm(adj_matrices[t], x_proj[t])
                    else:
                        agg = torch.zeros_like(x_proj[t])
                    out.append(agg.unsqueeze(0))
                x = torch.cat(out, dim=0) + b  # [T, N, H_out]

            if not is_last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GATEncoderVectorized(nn.Module):
    """Vectorized GAT encoder: precomputed edge indices + per-time-step attention aggregation.

    GAT attention depends on the node features at both ends of every edge, and the edge set
    differs across time steps, so it cannot be fully vectorized the way GCN can.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3, heads=4,
                 use_checkpoint=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.use_checkpoint = use_checkpoint

        self.convs = nn.ModuleList()
        current_dim = input_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                conv = GATConv(current_dim, output_dim, heads=1, concat=False, dropout=dropout)
                current_dim = output_dim
            else:
                conv = GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout)
                current_dim = hidden_dim * heads
            self.convs.append(conv)

    def _conv_with_checkpoint(self, conv, x, edge_index):
        """GATConv call with optional checkpointing (saves GPU memory during training)"""
        if self.use_checkpoint and self.training:
            # use_reentrant=False: recommended for PyTorch 2.0+
            return checkpoint(conv, x, edge_index, use_reentrant=False)
        return conv(x, edge_index)

    def forward(self, edge_index_list, node_feats):
        """edge_index_list: List[T] edge indices [2, E_t]; node_feats: [N, F], shared across all time steps. Returns [T, N, output_dim]."""
        T = len(edge_index_list)
        x_shared = node_feats  # [N, F]

        # First layer: node features are identical, but the edge sets differ -> per-time-step GAT
        first_out = []
        for t in range(T):
            h = self._conv_with_checkpoint(self.convs[0], x_shared, edge_index_list[t])
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            first_out.append(h.unsqueeze(0))
        x = torch.cat(first_out, dim=0)  # [T, N, H1]

        # Subsequent layers: the node embeddings already differ across time steps
        for layer_idx in range(1, self.num_layers):
            is_last = (layer_idx == self.num_layers - 1)
            out = []
            for t in range(T):
                h = self._conv_with_checkpoint(self.convs[layer_idx], x[t], edge_index_list[t])
                if not is_last:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                out.append(h.unsqueeze(0))
            x = torch.cat(out, dim=0)  # [T, N, H_next]

        return x
