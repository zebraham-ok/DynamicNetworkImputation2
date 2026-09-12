"""
GAT-GRU vectorized version (BiGRUImputationWithGATVectorized)
Pre-stores the edge_index list and uses GATEncoderVectorized.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from Models.common_vectorized import GATEncoderVectorized
from Models.common import build_feature_extractor, apply_node_feature_extractor


class BiGRUImputationWithGATVectorized(nn.Module):
    """GAT-GRU (Vectorized): precomputed edge indices + GATEncoderVectorized"""

    def __init__(self, dynamic_data: Data, static_hidden_dim, dynamic_hidden_dim,
                 num_gat_layers=2, num_rnn_layers=1, dropout=0.3,
                 time_steps=list(range(2013, 2026)), device=None, heads=4,
                 use_checkpoint=True,
                 use_fc_embedding=True, fc_embed_dim=128, fc_hidden_dim=None, fc_num_layers=3):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.time_steps = time_steps
        self.year_to_idx = {int(year): i for i, year in enumerate(time_steps)}
        self.num_nodes = dynamic_data.num_nodes
        node_feat_dim = dynamic_data.x.size(1)

        # Pre-store the edge_index list
        self.edge_index_list = []
        for t in time_steps:
            mask = dynamic_data.edge_time == t
            ei = dynamic_data.edge_index[:, mask].to(self.device)
            self.edge_index_list.append(ei)

        # Keep the raw node features
        self.raw_node_feats = dynamic_data.x.to(self.device)

        # Shared node-feature extractor (same module/hyper-parameters in every backbone model).
        # Input width = dense embedding (+ optional one-hot attribute blocks), output = fc_embed_dim.
        self.feature_extractor = build_feature_extractor(
            input_dim=node_feat_dim,
            enabled=use_fc_embedding,
            output_dim=fc_embed_dim,
            hidden_dim=fc_hidden_dim,
            dropout=dropout,
            num_layers=fc_num_layers,
        )
        encoder_input_dim = fc_embed_dim if self.feature_extractor is not None else node_feat_dim

        # GAT encoder
        self.static_encoder = GATEncoderVectorized(
            input_dim=encoder_input_dim,
            hidden_dim=static_hidden_dim,
            output_dim=static_hidden_dim,
            num_layers=num_gat_layers,
            dropout=dropout,
            heads=heads,
            use_checkpoint=use_checkpoint,
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
        static_sequence = self.static_encoder(self.edge_index_list, node_feats)

        # Bi-GRU temporal encoding
        gru_input = static_sequence.transpose(0, 1)  # [N, T, static_hidden_dim]
        dynamic_sequence, _ = self.temporal_encoder(gru_input)
        dynamic_embeddings = dynamic_sequence.transpose(0, 1)  # [T, N, 2*dynamic_hidden_dim]

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

        time_var = (time_positions.float() / num_timesteps).unsqueeze(1)

        pair_features = torch.cat([emb_u, emb_v, time_var], dim=-1)
        predictions = self.edge_predictor(pair_features).squeeze(-1)

        return predictions
