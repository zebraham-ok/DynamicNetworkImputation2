"""
Shared model components: FCEmbedding (feature dimension reduction) + GCNEncoder (graph encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FCEmbedding(nn.Module):
    """Three-layer fully-connected network that reduces high-dimensional sparse node features to a compact embedding"""
    def __init__(self, input_dim, embed_dim, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.fc(x)


class GCNEncoder(nn.Module):
    """GCN graph encoder, with an optional preceding FC dimension-reduction layer."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3,
                 use_fc_embedding=False, fc_embed_dim=128):
        super().__init__()
        self.use_fc_embedding = use_fc_embedding

        if use_fc_embedding:
            self.fc_embed = FCEmbedding(input_dim, fc_embed_dim, dropout)
            _input_dim = fc_embed_dim
        else:
            self.fc_embed = None
            _input_dim = input_dim

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(_input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.use_fc_embedding and self.fc_embed is not None:
            x = self.fc_embed(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
