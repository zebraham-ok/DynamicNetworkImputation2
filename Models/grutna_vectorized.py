"""
TNA (Temporal Network Aggregation) vectorized version (BiTNAImputationVectorized)
Precomputes the sparse normalized adjacency matrices, using torch.sparse.mm instead of GCNConv.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from Models.common_vectorized import precompute_adj_matrices
from Models.common import build_feature_extractor, apply_node_feature_extractor


class PyG_TemporalGCNLayerVectorized(nn.Module):
    """Vectorized version of the TNA core layer: GCN extracts spatial features -> GRU extracts temporal features -> Skip Connection."""

    def __init__(self, in_channels, out_channels, num_nodes, dropout=0.3, bidirectional=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.bidirectional = bidirectional
        self.out_channels = out_channels

        # Spatial convolution weights
        self.gcn_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.gcn_bias = nn.Parameter(torch.zeros(out_channels))
        self._reset_param(self.gcn_weight)

        self.gcn_norm = nn.LayerNorm(out_channels)

        # Temporal processing (GRU)
        self.rnn = nn.GRU(
            input_size=out_channels,
            hidden_size=out_channels,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Skip connection
        rnn_out_dim = out_channels * 2 if bidirectional else out_channels
        self.skip = nn.Linear(rnn_out_dim + out_channels, rnn_out_dim)
        self.norm = nn.LayerNorm(rnn_out_dim)

    def _reset_param(self, t):
        import math
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x_seq, adj_matrices):
        """x_seq: [T, N, in_channels]; adj_matrices: List[T] normalized adjacency matrices [N, N]. Returns [T, N, rnn_out_dim]."""
        num_times = x_seq.size(0)

        # Spatial dimension: the input differs per time step, so a per-time-step sparse.mm is needed
        x_proj = x_seq @ self.gcn_weight  # [T, N, out_channels]
        spatial_feats = []
        for t in range(num_times):
            if adj_matrices[t]._nnz() > 0:
                h_t = torch.sparse.mm(adj_matrices[t], x_proj[t])
            else:
                h_t = torch.zeros_like(x_proj[t])
            h_t = h_t + self.gcn_bias
            h_t = F.relu(h_t)
            h_t = self.gcn_norm(h_t)
            spatial_feats.append(h_t.unsqueeze(0))

        spatial_feats = torch.cat(spatial_feats, dim=0)  # [T, N, out_channels]

        # Temporal dimension: apply the GRU per node
        rnn_input = spatial_feats.transpose(0, 1)  # [N, T, out_channels]
        rnn_output, _ = self.rnn(rnn_input)  # [N, T, rnn_out_dim]

        # Fusion (Skip Connection)
        combined = torch.cat([rnn_output, spatial_feats.transpose(0, 1)], dim=-1)
        out = F.leaky_relu(self.skip(combined))
        out = self.norm(out)

        return out.transpose(0, 1)  # [T, N, rnn_out_dim]


class BiTNAImputationVectorized(nn.Module):
    """TNA (Vectorized): precomputed adjacency matrices + torch.sparse.mm instead of GCNConv"""

    def __init__(self, dynamic_data, static_hidden_dim, dynamic_hidden_dim,
                 dropout=0.3, time_steps=list(range(2013, 2026)), device=None,
                 use_fc_embedding=True, fc_embed_dim=128, fc_hidden_dim=None, fc_num_layers=3):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.time_steps = time_steps
        self.year_to_idx = {int(year): i for i, year in enumerate(time_steps)}
        self.num_nodes = dynamic_data.num_nodes
        self.node_feat_dim = dynamic_data.x.size(1)

        # Precompute the sparse adjacency matrices
        self.adj_matrices = precompute_adj_matrices(
            dynamic_data, time_steps, self.num_nodes, self.device
        )
        # Keep the raw node features
        self.raw_node_feats = dynamic_data.x.to(self.device)

        # Shared node-feature extractor (same module/hyper-parameters in every backbone model)
        self.feature_extractor = build_feature_extractor(
            input_dim=self.node_feat_dim,
            enabled=use_fc_embedding,
            output_dim=fc_embed_dim,
            hidden_dim=fc_hidden_dim,
            dropout=dropout,
            num_layers=fc_num_layers,
        )
        tna_input_dim = fc_embed_dim if self.feature_extractor is not None else self.node_feat_dim

        # Two temporal graph convolution layers
        self.tna_layer1 = PyG_TemporalGCNLayerVectorized(
            tna_input_dim, static_hidden_dim, self.num_nodes, dropout, bidirectional=True
        )

        # Second layer: the input dimension is the output of layer1 (static_hidden_dim * 2)
        self.tna_layer2 = PyG_TemporalGCNLayerVectorized(
            static_hidden_dim * 2, dynamic_hidden_dim, self.num_nodes, dropout, bidirectional=True
        )

        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(4 * dynamic_hidden_dim + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.to(self.device)

    def forward(self, node_pairs, time_indices):
        """node_pairs: [B, 2] node pairs; time_indices: [B] time-step years. Returns the probability that each edge exists [B]."""
        num_times = len(self.time_steps)

        # Project the (time-invariant) node features once through the shared extractor
        node_feats = apply_node_feature_extractor(self.feature_extractor, self.raw_node_feats)

        # x_seq: [T, N, node_feat_dim], all time steps share the same node features
        x_seq = node_feats.unsqueeze(0).expand(num_times, -1, -1)

        z = self.tna_layer1(x_seq, self.adj_matrices)
        z = F.dropout(z, p=0.3, training=self.training)

        # Layer 2: z shape [T, N, dynamic_hidden_dim * 2]
        dynamic_embeddings = self.tna_layer2(z, self.adj_matrices)

        # Batch edge prediction
        time_positions = []
        for t_idx in time_indices:
            pos = self.year_to_idx.get(int(t_idx.item()), -1)
            time_positions.append(pos)

        time_positions = torch.tensor(time_positions, device=self.device, dtype=torch.long)

        valid_mask = time_positions >= 0
        if not valid_mask.any():
            return torch.tensor([], device=self.device)

        time_positions = time_positions[valid_mask]
        node_pairs_valid = node_pairs[valid_mask]

        u_indices = node_pairs_valid[:, 0].long()
        v_indices = node_pairs_valid[:, 1].long()

        emb_u = dynamic_embeddings[time_positions, u_indices]
        emb_v = dynamic_embeddings[time_positions, v_indices]

        time_var = (time_positions.float() / num_times).unsqueeze(1)

        pair_features = torch.cat([emb_u, emb_v, time_var], dim=-1)
        predictions = self.edge_predictor(pair_features).squeeze(-1)

        return predictions
