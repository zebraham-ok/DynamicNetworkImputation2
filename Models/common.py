"""
Shared model components:
    - NodeFeatureExtractor: the SHARED node-feature extractor used by every backbone model
    - build_feature_extractor / apply_node_feature_extractor: the single construction /
      application entry points, so no model can silently skip the projection
    - GCNEncoder: GCN graph encoder (non-vectorised EvolveGCN path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NodeFeatureExtractor(nn.Module):
    """Backbone-agnostic projection of the raw node feature matrix X.

    All backbone models (GAT-GRU, Node-GRU, TNA, EvolveGCN, Temp-SEAL) share this exact
    module and the exact same hyper-parameters, so the representation entering their own
    spatial encoder is identical.  It is only a genuine dimensionality reduction when the
    input width exceeds ``output_dim``, which is the case when X is the dense text embedding
    concatenated with the optional one-hot attribute blocks
    (``country`` / ``industry_2nd`` / ``category_3rd``, see ``Data/company_dataset.py``).

    Structure (``num_layers`` linear layers):
        num_layers=1:  Linear(in, out)
        num_layers>=2: [Linear(in, hidden) - ReLU - Dropout] x (num_layers-1) then Linear(hidden, out)
    """

    def __init__(self, input_dim, output_dim=128, hidden_dim=None, dropout=0.3, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = int(hidden_dim) if hidden_dim else output_dim * 2
        self.hidden_dim = hidden_dim
        self.num_layers = int(num_layers)

        if self.num_layers <= 1:
            self.fc = nn.Sequential(nn.Linear(input_dim, output_dim))
        else:
            layers = []
            in_dim = input_dim
            for _ in range(self.num_layers - 1):
                layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, output_dim))
            self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


def build_feature_extractor(input_dim, enabled=True, output_dim=128, hidden_dim=None,
                            dropout=0.3, num_layers=3):
    """Single factory for the shared node-feature extractor.

    Every model must build its extractor through this function so that a change of the
    architecture is applied uniformly instead of per model.
    """
    if not enabled:
        return None
    return NodeFeatureExtractor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
    )


def apply_node_feature_extractor(extractor, x):
    """Apply the shared extractor (no-op when it is disabled)."""
    if extractor is None:
        return x
    return extractor(x)


class FCEmbedding(nn.Module):
    """Legacy three-layer fully-connected dimension reducer, kept for backward compatibility.

    Superseded by :class:`NodeFeatureExtractor`; new code must use the shared extractor.
    """
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
    """GCN graph encoder (non-vectorised EvolveGCN path)."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
