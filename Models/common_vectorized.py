"""
Shared vectorized components:
    - precompute_adj_matrices: precompute the normalized sparse adjacency matrices for all time steps
    - GCNEncoderVectorized: vectorized GCN encoder, uses torch.sparse.mm instead of GCNConv
    - GATEncoderVectorized: vectorized GAT encoder, precomputes edge indices to avoid rebuilding Data objects

Note: the node-feature projection is NOT done here any more.  Every backbone model owns one
shared ``Models.common.NodeFeatureExtractor`` (see ``build_feature_extractor``) and feeds the
projected features into these encoders, so the extractor cannot differ across models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv
from Models.common import build_feature_extractor, apply_node_feature_extractor  # noqa: F401


# Aggregation-direction vocabulary shared by every hand-written GCN path. It mirrors
# GATEncoderVectorized.MESSAGE_DIRECTIONS below and the same-named key in
# Training/common_config.yaml (model.kwargs.message_direction).
MESSAGE_DIRECTIONS = ('source_to_target', 'target_to_source')


def precompute_adj_matrices(dynamic_data, time_steps, num_nodes, device,
                            message_direction='source_to_target'):
    """Precompute the normalized sparse adjacency matrices A_norm = D^{-1/2} A D^{-1/2} for all time steps; returns List[sparse_coo_tensor].

    ``message_direction`` is the shared aggregation-direction switch
    (``Training/common_config.yaml -> model.kwargs.message_direction``), the same value the GAT
    path forwards to ``GATConv(flow=...)``:

        'source_to_target' (default since 2026-09-16)
            messages travel supplier -> customer, so a node aggregates its SUPPLIERS
            (in-neighbours / the upstream side). Formally ``out[i] = sum_j A[i, j] x[j]`` with
            ``i = target``, i.e. the textbook GCN normalisation (PyG's own ``gcn_norm`` uses the
            in-degree the same way).
        'target_to_source'
            messages travel customer -> supplier, so a node aggregates its CUSTOMERS
            (out-neighbours / the downstream side). This is what this module did before
            2026-09-16 and is kept ONLY to reproduce that historical ablation.

    Only which endpoint of a directed edge is updated changes; the produced matrices keep the
    same shape (N, N), so every model reads them unchanged and the checkpoints stay comparable.
    """
    if message_direction not in MESSAGE_DIRECTIONS:
        raise ValueError(
            f"message_direction must be one of {MESSAGE_DIRECTIONS}, got {message_direction!r}"
        )

    adj_matrices = []
    for t in time_steps:
        mask = dynamic_data.edge_time == t
        edge_index = dynamic_data.edge_index[:, mask]

        if edge_index.size(1) > 0:
            row, col = edge_index  # row = supplier (source), col = customer (target)
            if message_direction == 'source_to_target':
                # Aggregate in-neighbours: the TARGET collects the messages of its SUPPLIERS
                walk_row, walk_col = col, row
            else:
                # Historical behaviour: the SOURCE collects the messages of its CUSTOMERS
                walk_row, walk_col = row, col

            deg = torch.bincount(walk_row, minlength=num_nodes).float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[walk_row] * deg_inv_sqrt[walk_col]

            adj = torch.sparse_coo_tensor(
                torch.stack([walk_row, walk_col]), norm,
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Store the weights and biases of each layer, replacing GCNConv
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        if num_layers == 1:
            layer_dims = [input_dim, output_dim]
        else:
            layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            self.weights.append(nn.Parameter(torch.Tensor(layer_dims[i], layer_dims[i+1])))
            self.biases.append(nn.Parameter(torch.zeros(layer_dims[i+1])))
            self._reset_param(self.weights[-1])

    def _reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, adj_matrices, node_feats):
        """adj_matrices: List[T] adjacency matrices [N, N]; node_feats: [N, input_dim] (already projected), shared across all time steps. Returns [T, N, output_dim]."""
        T = len(adj_matrices)
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

    ``message_direction`` (PyG's ``flow`` argument, exposed as ``model.kwargs.message_direction``)
    decides which endpoint of a directed edge is the one being updated:

        'source_to_target' (default, historical GAT-GRU behaviour)
            messages travel supplier -> customer, so a node aggregates its SUPPLIERS
            (= its in-neighbours / the upstream side).
        'target_to_source'
            messages travel customer -> supplier, so a node aggregates its CUSTOMERS
            (= its out-neighbours / the downstream side). This is the direction the
            hand-written GCN path (``precompute_adj_matrices`` + ``torch.sparse.mm``, used by
            Node-GRU / TNA / EGCN) has always used.

    Only the direction of the message passing changes - the parameter shapes and therefore the
    checkpoint layout stay identical, so the two settings are directly comparable.
    """
    MESSAGE_DIRECTIONS = ('source_to_target', 'target_to_source')

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3, heads=4,
                 use_checkpoint=True, message_direction='source_to_target'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.use_checkpoint = use_checkpoint
        if message_direction not in self.MESSAGE_DIRECTIONS:
            raise ValueError(
                f"message_direction must be one of {self.MESSAGE_DIRECTIONS}, got {message_direction!r}"
            )
        self.message_direction = message_direction

        self.convs = nn.ModuleList()
        current_dim = input_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                conv = GATConv(current_dim, output_dim, heads=1, concat=False, dropout=dropout,
                               flow=message_direction)
                current_dim = output_dim
            else:
                conv = GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout,
                               flow=message_direction)
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
