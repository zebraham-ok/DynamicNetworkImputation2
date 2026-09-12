"""
Node-GRU vectorized version (BiGRUImputationVectorized)
Precomputes the sparse normalized adjacency matrices, using torch.sparse.mm instead of PyG GCNConv.
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data

from Models.common_vectorized import precompute_adj_matrices, GCNEncoderVectorized
from Models.common import build_feature_extractor, apply_node_feature_extractor


class BiGRUImputationVectorized(nn.Module):
    """Node-GRU (Vectorized): precomputed adjacency matrices + torch.sparse.mm"""

    def __init__(self, dynamic_data: Data, static_hidden_dim, dynamic_hidden_dim,
                 num_gcn_layers=2, num_rnn_layers=1, dropout=0.3,
                 time_steps=list(range(2013, 2026)), device=None,
                 use_fc_embedding=True, fc_embed_dim=128, fc_hidden_dim=None, fc_num_layers=3):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.time_steps = time_steps
        self.year_to_idx = {int(year): i for i, year in enumerate(time_steps)}
        self.num_nodes = dynamic_data.num_nodes
        node_feat_dim = dynamic_data.x.size(1)

        # Precompute the sparse adjacency matrices
        self.adj_matrices = precompute_adj_matrices(
            dynamic_data, time_steps, self.num_nodes, self.device
        )
        # Keep the raw node features
        self.raw_node_feats = dynamic_data.x.to(self.device)

        # Shared node-feature extractor (same module/hyper-parameters in every backbone model)
        self.feature_extractor = build_feature_extractor(
            input_dim=node_feat_dim,
            enabled=use_fc_embedding,
            output_dim=fc_embed_dim,
            hidden_dim=fc_hidden_dim,
            dropout=dropout,
            num_layers=fc_num_layers,
        )
        encoder_input_dim = fc_embed_dim if self.feature_extractor is not None else node_feat_dim

        # Vectorized GCN encoder
        self.static_encoder = GCNEncoderVectorized(
            input_dim=encoder_input_dim,
            hidden_dim=static_hidden_dim,
            output_dim=static_hidden_dim,
            num_layers=num_gcn_layers,
            dropout=dropout
        )

        # Dynamic temporal encoder (Bi-GRU)
        self.temporal_encoder = nn.GRU(
            input_size=static_hidden_dim,
            hidden_size=dynamic_hidden_dim,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0
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

        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.static_hidden_dim = static_hidden_dim
        self.to(self.device)

    def forward(self, node_pairs, time_indices):
        """node_pairs: [B, 2] node pairs; time_indices: [B] time-step years. Returns the probability that each edge exists [B]."""
        num_timesteps = len(self.time_steps)

        # Project the (time-invariant) node features once through the shared extractor
        node_feats = apply_node_feature_extractor(self.feature_extractor, self.raw_node_feats)

        # static_sequence: [T, N, static_hidden_dim]
        static_sequence = self.static_encoder(self.adj_matrices, node_feats)

        # Bi-GRU temporal encoding: [N, T, static_hidden_dim] -> [N, T, 2*dynamic_hidden_dim]
        gru_input = static_sequence.transpose(0, 1)
        dynamic_sequence, _ = self.temporal_encoder(gru_input)
        dynamic_embeddings = dynamic_sequence.transpose(0, 1)  # [T, N, 2*dynamic_hidden_dim]

        # Convert years to time-step indices
        time_positions = []
        for t_idx in time_indices:
            pos = self.year_to_idx.get(int(t_idx.item()), -1)
            time_positions.append(pos)

        time_positions = torch.tensor(time_positions, device=self.device, dtype=torch.long)

        # Filter out invalid positions
        valid_mask = time_positions >= 0
        if not valid_mask.any():
            return torch.tensor([], device=self.device)

        time_positions = time_positions[valid_mask]
        node_pairs_valid = node_pairs[valid_mask]

        u_indices = node_pairs_valid[:, 0].long()
        v_indices = node_pairs_valid[:, 1].long()

        # Batch-fetch node embeddings [B, 2*dynamic_hidden_dim]
        emb_u = dynamic_embeddings[time_positions, u_indices]
        emb_v = dynamic_embeddings[time_positions, v_indices]

        time_var = (time_positions.float() / num_timesteps).unsqueeze(1)

        pair_features = torch.cat([emb_u, emb_v, time_var], dim=-1)
        predictions = self.edge_predictor(pair_features).squeeze(-1)

        return predictions
