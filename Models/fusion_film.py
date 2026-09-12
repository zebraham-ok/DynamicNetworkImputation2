"""
Fusion option 4 -- FiLM modulation (single branch, cheapest + clearest mechanism story).

Design doc: ``results/FUSION_DESIGN_OPTIONS_0912.md`` section 8.

Instead of letting the evolved operator ``W_t`` rewrite the representation (option 3), option 4
only lets it produce **a pair of modulation parameters per year** that scale / shift the attention
output:

    z    = feature_extractor(X)              [N, 128]      shared, time-invariant
    s_t  = GAT_t(z)                          [T, N, 128]   attention aggregation per year
    g_t  = global drift summary of year t    [128]
    (gamma_t, beta_t) = MLP(g_t)             [2 x 128]
    s'_t = (1 + gamma_t) * s_t + beta_t      [T, N, 128]   FiLM modulation
    d    = BiGRU(s')                         [T, N, 2*64]  per-firm 13-year trajectory
    p    = head([d_u, d_v, t/T])             [B]

The paper-facing sentence is the mechanism: **system-level operator drift (gamma_t, beta_t)
modulates the pair-wise representation at every time step** -- it adds the global information
GAT-GRU is missing without destroying its per-firm trajectory structure.

Two design choices worth knowing:

* ``summary_mode`` decides where ``g_t`` comes from.
  - ``weights`` (default): evolve ``W_t`` with the same ``MatGRUCell_PyG`` as EvolveGCN-H and take
    ``g_t = mean_dim0(W_t)`` (== per-column mean, so the width is always ``hidden_dim`` even when
    the driver width differs).  This is the faithful "operator drift" reading and costs one
    MatGRU (+0.163 M at 128x128).
  - ``attention``: ``g_t = mean_N(s_t)``.  Costs nothing extra and is still time-varying (the
    edge set ``A_t`` changes every year), but there is no explicit operator any more.

  A third candidate -- summarising the extractor output ``mean_N(z)`` -- is deliberately NOT
  offered: the shared extractor is applied once and the same ``z`` feeds every year, so that
  summary would be constant over ``t`` and the "per-time-step modulation" mechanism would vanish.

* The last MLP layer is **zero-initialised**, so at initialisation ``gamma = beta = 0`` and
  ``s'_t = s_t`` exactly.  FiLM therefore starts life as the plain GAT-GRU baseline and can only
  move away from it if that helps: useful for a clean "modulation adds X" ablation and for
  avoiding the "new architecture is worse at init" failure mode.

Module naming follows the two-stage freeze contract hard-coded in
``Data/company_dataset.load_pretrained_backbone`` / ``reinit_trainable_parts``:

    feature_extractor   shared node-feature extractor   Phase 2: FROZEN
    static_encoder      GAT attention                   Phase 2: FROZEN (and saved into backbone.pth)
    temporal_encoder    drift summary + FiLM + BiGRU    Phase 2: re-initialised and trained
    edge_predictor      pair head                       Phase 2: re-initialised and trained
"""

import math

import torch
import torch.nn as nn

from Models.common import build_feature_extractor, apply_node_feature_extractor
from Models.common_vectorized import GATEncoderVectorized
from Models.egcn import MatGRUCell_PyG

SUMMARY_MODES = ('weights', 'attention')


class GlobalDriftSummarizer(nn.Module):
    """Turns the global drift of year t into a FiLM (gamma_t, beta_t) pair.

    ``summary_mode='weights'``   g_t = mean_dim0(W_t),  W_t = MatGRU(W_{t-1}, mean_N(z))
    ``summary_mode='attention'`` g_t = mean_N(s_t)
    """

    def __init__(self, driver_dim, hidden_dim, summary_mode='weights', mlp_hidden=256,
                 dropout=0.3):
        super().__init__()
        if summary_mode not in SUMMARY_MODES:
            raise ValueError(f"summary_mode must be one of {SUMMARY_MODES}, got {summary_mode!r}")
        self.driver_dim = int(driver_dim)
        self.hidden_dim = int(hidden_dim)
        self.summary_mode = summary_mode
        self.uses_operator = (summary_mode == 'weights')

        if self.uses_operator:
            # Same operator evolution as EvolveGCN-H (Models/egcn.py :: GRCU_PyG_Vectorized)
            self.evolve_weights = MatGRUCell_PyG(rows=self.driver_dim, cols=self.hidden_dim)
            self.gcn_init_weights = nn.Parameter(torch.Tensor(self.driver_dim, self.hidden_dim))
            self.reset_param(self.gcn_init_weights)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 2 * self.hidden_dim),
        )
        # Zero-init => (gamma, beta) = 0 at init => the model starts as plain GAT-GRU.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def reset_param(self, t):
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, static_sequence, driver_features):
        """static_sequence: [T, N, hidden_dim]; driver_features: [N, driver_dim] (time-invariant).

        Returns (gamma, beta) each of shape [T, hidden_dim].
        """
        num_timesteps = static_sequence.size(0)
        gamma_list, beta_list = [], []

        if self.uses_operator:
            weights = self.gcn_init_weights
            # mean_N(z) is time-invariant, so detaching once keeps the whole operator branch out
            # of the attention branch's backward graph (same rationale as option 3).
            driver = driver_features.detach()

        for t in range(num_timesteps):
            if self.uses_operator:
                weights = self.evolve_weights(weights, driver)
                summary = weights.mean(dim=0)              # [hidden_dim] (per-column mean)
            else:
                summary = static_sequence[t].mean(dim=0)   # [hidden_dim]

            gamma_beta = self.mlp(summary)                 # [2 * hidden_dim]
            gamma_list.append(gamma_beta[:self.hidden_dim].unsqueeze(0))
            beta_list.append(gamma_beta[self.hidden_dim:].unsqueeze(0))

        return torch.cat(gamma_list, dim=0), torch.cat(beta_list, dim=0)


class FiLMEncoder(nn.Module):
    """Global-drift FiLM modulation + per-firm BiGRU trajectory encoder (the trainable temporal stack)."""

    def __init__(self, driver_dim, hidden_dim, dynamic_hidden_dim, num_rnn_layers=1,
                 dropout=0.3, summary_mode='weights', mlp_hidden=256, unit_offset=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.unit_offset = bool(unit_offset)

        self.summarizer = GlobalDriftSummarizer(
            driver_dim=driver_dim,
            hidden_dim=hidden_dim,
            summary_mode=summary_mode,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=dynamic_hidden_dim,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0,
        )

    def forward(self, static_sequence, driver_features):
        """static_sequence: [T, N, hidden_dim].

        Returns ([T, N, 2 * dynamic_hidden_dim], gamma, beta, modulated) where ``modulated`` is the
        FiLM output s'_t (== ``static_sequence`` at initialisation, see the module docstring).
        """
        gamma, beta = self.summarizer(static_sequence, driver_features)  # [T, hidden_dim] each

        gain = (1.0 + gamma) if self.unit_offset else gamma
        modulated = gain.unsqueeze(1) * static_sequence + beta.unsqueeze(1)  # [T, N, hidden_dim]

        gru_input = modulated.transpose(0, 1)                                # [N, T, hidden_dim]
        dynamic_sequence, _ = self.gru(gru_input)
        return dynamic_sequence.transpose(0, 1), gamma, beta, modulated      # [T, N, 2 * dyn], ...


class FiLMGATGRU(nn.Module):
    """GAT-GRU whose attention output is FiLM-modulated by the system-level operator drift.

    ``forward(link_indices, current_times)`` -- the standard 5-backbone signature, so the trainer
    needs no change (``_forward_takes_static_graph`` sees no ``data`` argument).
    """

    def __init__(self, dynamic_data, static_hidden_dim, dynamic_hidden_dim,
                 num_gat_layers=2, num_rnn_layers=1, dropout=0.3,
                 time_steps=list(range(2013, 2026)), device=None, heads=4,
                 use_checkpoint=True,
                 use_fc_embedding=True, fc_embed_dim=128, fc_hidden_dim=None, fc_num_layers=3,
                 summary_mode='weights', mlp_hidden=256, film_unit_offset=True):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.time_steps = time_steps
        self.year_to_idx = {int(year): i for i, year in enumerate(time_steps)}
        self.num_nodes = dynamic_data.num_nodes
        node_feat_dim = dynamic_data.x.size(1)

        # Pre-store the edge_index list (one entry per year)
        self.edge_index_list = []
        for t in time_steps:
            mask = dynamic_data.edge_time == t
            self.edge_index_list.append(dynamic_data.edge_index[:, mask].to(self.device))

        self.raw_node_feats = dynamic_data.x.to(self.device)

        # Shared node-feature extractor (same module / hyper-parameters in every backbone model)
        self.feature_extractor = build_feature_extractor(
            input_dim=node_feat_dim,
            enabled=use_fc_embedding,
            output_dim=fc_embed_dim,
            hidden_dim=fc_hidden_dim,
            dropout=dropout,
            num_layers=fc_num_layers,
        )
        encoder_input_dim = fc_embed_dim if self.feature_extractor is not None else node_feat_dim

        # GAT attention branch: consumes z and produces s_t = GAT_t(z), exactly as in GAT-GRU.
        self.static_encoder = GATEncoderVectorized(
            input_dim=encoder_input_dim,
            hidden_dim=static_hidden_dim,
            output_dim=static_hidden_dim,
            num_layers=num_gat_layers,
            dropout=dropout,
            heads=heads,
            use_checkpoint=use_checkpoint,
        )

        # Temporal stack: global drift summary -> FiLM modulation -> per-firm BiGRU.
        self.temporal_encoder = FiLMEncoder(
            driver_dim=encoder_input_dim,
            hidden_dim=static_hidden_dim,
            dynamic_hidden_dim=dynamic_hidden_dim,
            num_rnn_layers=num_rnn_layers,
            dropout=dropout,
            summary_mode=summary_mode,
            mlp_hidden=mlp_hidden,
            unit_offset=film_unit_offset,
        )

        # Pair head: identical width to GAT-GRU's ([d_u, d_v, t] = 4 * hidden + 1)
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
        # Filled by every forward(): detached (gamma, beta) and the FiLM output of the last pass,
        # for analysis / plots (gamma_t over t is the interpretability hook of this design).
        self.last_gamma = None
        self.last_beta = None
        self.last_modulated = None
        self.to(self.device)

    def get_last_modulation(self):
        """Detached (gamma, beta) of the most recent forward pass, each [T, static_hidden_dim]."""
        return self.last_gamma, self.last_beta

    def get_last_modulated(self):
        """Detached FiLM output s' of the most recent forward pass, [T, N, static_hidden_dim]."""
        return self.last_modulated

    def forward(self, node_pairs, time_indices):
        """node_pairs: [B, 2] node pairs; time_indices: [B] years. Returns p(edge exists) [B]."""
        num_timesteps = len(self.time_steps)

        # Project the (time-invariant) node features once through the shared extractor
        node_feats = apply_node_feature_extractor(self.feature_extractor, self.raw_node_feats)

        # static_sequence: [T, N, static_hidden_dim]
        static_sequence = self.static_encoder(self.edge_index_list, node_feats)

        # FiLM modulation + temporal encoding: [T, N, 2 * dynamic_hidden_dim]
        dynamic_embeddings, gamma, beta, modulated = self.temporal_encoder(static_sequence, node_feats)
        self.last_gamma = gamma.detach()
        self.last_beta = beta.detach()
        self.last_modulated = modulated.detach()

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
