"""
Fusion option 3 -- temporal injection (single branch).

Design doc: ``results/FUSION_DESIGN_OPTIONS_0912.md`` section 7.

EvolveGCN-H's globally evolved operator ``W_t`` is demoted from "the output of a second
model" to "one transformation layer inside GAT-GRU", so the two information paths meet
*serially* instead of in the pair head:

    z    = feature_extractor(X)                 [N, 128]      shared, time-invariant
    s_t  = GAT_t(z)                             [T, N, 128]   attention aggregation per year
    W_t  = MatGRU(W_{t-1}, mean_N(z))           [128, 128]    EvolveGCN-H operator evolution
    u_t  = act(s_t @ W_t + A_t . (s_t @ W_t))   [T, N, 128]   the injection point
    d    = BiGRU(u)                             [T, N, 2*64]  per-firm 13-year trajectory
    p    = head([d_u, d_v, t/T])                [B]

Two consequences of the shared extractor being static (``Models/common.NodeFeatureExtractor``
is applied once, so the same ``z`` feeds every year) are worth stating explicitly:

* ``mean_N(z)`` is **time-invariant**.  ``W_t`` therefore evolves as a function of ``t`` through
  the MatGRU recurrence, not because its input changes.  This is exactly how ``EvolveGCN-H``
  behaves in ``Models/egcn.py`` (``node_feats_seq`` is an ``expand`` of one tensor), so option 3
  inherits the operator dynamics rather than inventing new ones.
* ``detach_global`` (default ``True``) cuts the gradient of the driver branch, so the operator
  gradient cannot compete with the attention branch for the same representation (design doc
  section 7, risk "gradient coupling").
* ``residual_injection`` (default ``True``) makes the injection **residual** with a
  zero-initialised per-channel gate:  ``u_t = s_t + g * act(A_t (s_t W_t) + s_t W_t)`` with
  ``g = 0`` at init, hence ``u_t == s_t`` **exactly** (in train and eval mode) and the encoder
  starts life as plain GAT-GRU.  This is the exact analogue of option 4's zero-initialised FiLM
  MLP (``gamma = beta = 0 => s' == s``) and enforces the same "a new architecture must not be
  worse than the backbone at initialisation" contract.  The gradient w.r.t. ``g`` is non-zero at
  init (``g=0`` does not zero the gate's gradient), so the injection can still creep in on its
  own schedule.  Set ``residual_injection: false`` in the config to recover the original
  non-residual form ``u_t = act(A_t (s_t W_t) + s_t W_t)``: that variant starts from a
  near-constant output (all samples on the same side of the 0.5 boundary) and is kept only for
  comparison.

The operator evolution is **not** re-implemented: ``MatGRUCell_PyG`` and
``precompute_adj_matrices`` are imported from ``Models/egcn.py`` and ``Models/common_vectorized.py``
so a future change to the EvolveGCN-H recipe propagates to this model automatically.

Module naming follows the two-stage freeze contract hard-coded in
``Data/company_dataset.load_pretrained_backbone`` / ``reinit_trainable_parts``:

    feature_extractor   shared node-feature extractor   Phase 2: FROZEN
    static_encoder      GAT attention                   Phase 2: FROZEN (and saved into backbone.pth)
    temporal_encoder    MatGRU operator + BiGRU         Phase 2: re-initialised and trained
    edge_predictor      pair head                       Phase 2: re-initialised and trained

The MatGRU lives inside ``temporal_encoder`` on purpose (design doc section 11-2): that makes
Phase 2 restart *every* temporal component of option 3 from scratch on top of a frozen GAT
backbone, which is the only way its frozen-mode numbers are comparable with GAT-GRU's.
"""

import math

import torch
import torch.nn as nn

from Models.common import build_feature_extractor, apply_node_feature_extractor
from Models.common_vectorized import GATEncoderVectorized, precompute_adj_matrices
from Models.egcn import MatGRUCell_PyG


class TemporalInjectionEncoder(nn.Module):
    """Evolved operator + adjacency injection + per-firm BiGRU trajectory encoder.

    This is the whole trainable temporal stack, packed into one module so that
    ``reinit_trainable_parts()`` resets it as a unit (see the module docstring).
    """

    def __init__(self, driver_dim, hidden_dim, dynamic_hidden_dim, num_rnn_layers=1,
                 dropout=0.3, activation=None, detach_global=True, use_adjacency=True,
                 residual_injection=True):
        super().__init__()
        self.driver_dim = int(driver_dim)
        self.hidden_dim = int(hidden_dim)
        self.detach_global = bool(detach_global)
        self.use_adjacency = bool(use_adjacency)
        self.residual_injection = bool(residual_injection)
        self.activation = activation if activation is not None else nn.ReLU()

        if self.residual_injection:
            # Per-channel injection gate, zero => u_t == s_t at init (encoder == plain GAT-GRU).
            # Non-zero gradient at g=0, so training can open the gate on its own.
            self.injection_gate = nn.Parameter(torch.zeros(self.hidden_dim))

        # EvolveGCN-H's operator evolution, reused verbatim (Models/egcn.py :: GRCU_PyG_Vectorized).
        self.evolve_weights = MatGRUCell_PyG(rows=self.driver_dim, cols=self.hidden_dim)
        self.gcn_init_weights = nn.Parameter(torch.Tensor(self.driver_dim, self.hidden_dim))
        self.reset_param(self.gcn_init_weights)

        # Per-firm 13-year trajectory, identical in shape to GAT-GRU's temporal_encoder.
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=dynamic_hidden_dim,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def reset_param(self, t):
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, adj_matrices, static_sequence, driver_features):
        """adj_matrices: List[T] normalised sparse A_t; static_sequence: [T, N, hidden_dim];
        driver_features: [N, driver_dim] (the shared extractor output z, time-invariant).

        Returns the per-year node representation [T, N, 2 * dynamic_hidden_dim].
        """
        weights = self.gcn_init_weights
        # The driver is constant over t, so detaching once is enough and keeps the whole
        # operator branch out of the attention branch's backward graph.
        driver = driver_features.detach() if self.detach_global else driver_features

        injected = []
        for t in range(static_sequence.size(0)):
            weights = self.evolve_weights(weights, driver)

            transformed = static_sequence[t] @ weights  # [N, hidden_dim]
            if self.use_adjacency and adj_matrices[t]._nnz() > 0:
                out = torch.sparse.mm(adj_matrices[t], transformed) + transformed
            else:
                out = transformed

            if self.residual_injection:
                # u_t = s_t + g * (A_t (s_t W_t) + s_t W_t), with the per-channel gate g zero-init
                # => u_t == s_t at init, so the encoder is bit-identical to GAT-GRU.  The correction
                # is kept LINEAR (no `self.activation`): on the real graph the projected residual
                # A_t (s_t W_t) + s_t W_t is mean-zero at ~1e-4 scale, so an EGCN-style ReLU would
                # leave 97.4% of its entries at exactly 0 and kill both the injected signal and the
                # gradient of g / W_t.  Non-linearity is provided downstream by the BiGRU.
                # Dropout is applied to the correction only, so the identity path stays clean.
                injected.append((static_sequence[t]
                                 + self.injection_gate * self.dropout(out)).unsqueeze(0))
            else:
                injected.append(self.dropout(self.activation(out)).unsqueeze(0))

        injected = torch.cat(injected, dim=0)      # [T, N, hidden_dim]
        gru_input = injected.transpose(0, 1)       # [N, T, hidden_dim]
        dynamic_sequence, _ = self.gru(gru_input)
        return dynamic_sequence.transpose(0, 1)    # [T, N, 2 * dynamic_hidden_dim]


class TemporalInjectionGATGRU(nn.Module):
    """GAT-GRU with EvolveGCN-H's evolved operator injected between attention and the BiGRU.

    ``forward(link_indices, current_times)`` -- the standard 5-backbone signature, so the trainer
    needs no change (``_forward_takes_static_graph`` sees no ``data`` argument).
    """

    def __init__(self, dynamic_data, static_hidden_dim, dynamic_hidden_dim,
                 num_gat_layers=2, num_rnn_layers=1, dropout=0.3,
                 time_steps=list(range(2013, 2026)), device=None, heads=4,
                 use_checkpoint=True, message_direction='source_to_target',
                 use_fc_embedding=True, fc_embed_dim=128, fc_hidden_dim=None, fc_num_layers=3,
                 activation=None, detach_global=True, use_adjacency=True,
                 residual_injection=True):
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

        # W_t is driven by mean_N(z) (width encoder_input_dim) AND applied to the GAT output
        # (width static_hidden_dim), so the two widths must coincide here.  FiLM (option 4) has no
        # such constraint because it only reads statistics out of W_t.
        if int(static_hidden_dim) != int(encoder_input_dim):
            raise ValueError(
                f"TemporalInjectionGATGRU requires static_hidden_dim == the extractor output width "
                f"(fc_embed_dim), got static_hidden_dim={static_hidden_dim} vs "
                f"encoder_input_dim={encoder_input_dim}. Set static_hidden_dim={encoder_input_dim} "
                f"in Models/configs/fusion_temporal_injection.yaml."
            )

        # GAT attention branch: consumes z and produces s_t = GAT_t(z), exactly as in GAT-GRU.
        self.static_encoder = GATEncoderVectorized(
            input_dim=encoder_input_dim,
            hidden_dim=static_hidden_dim,
            output_dim=static_hidden_dim,
            num_layers=num_gat_layers,
            dropout=dropout,
            heads=heads,
            use_checkpoint=use_checkpoint,
            message_direction=message_direction,
        )

        # A_t for the second (injection) aggregation, through the SAME implementation as
        # EvolveGCN-H rather than a private copy of the normalisation.
        self.adj_matrices = precompute_adj_matrices(
            dynamic_data, time_steps, self.num_nodes, self.device
        )

        # Temporal stack: evolve the operator on z, inject it into the attention output, then GRU.
        self.temporal_encoder = TemporalInjectionEncoder(
            driver_dim=encoder_input_dim,
            hidden_dim=static_hidden_dim,
            dynamic_hidden_dim=dynamic_hidden_dim,
            num_rnn_layers=num_rnn_layers,
            dropout=dropout,
            activation=activation,
            detach_global=detach_global,
            use_adjacency=use_adjacency,
            residual_injection=residual_injection,
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
        self.to(self.device)

    def forward(self, node_pairs, time_indices):
        """node_pairs: [B, 2] node pairs; time_indices: [B] years. Returns p(edge exists) [B]."""
        num_timesteps = len(self.time_steps)

        # Project the (time-invariant) node features once through the shared extractor
        node_feats = apply_node_feature_extractor(self.feature_extractor, self.raw_node_feats)

        # static_sequence: [T, N, static_hidden_dim]
        static_sequence = self.static_encoder(self.edge_index_list, node_feats)

        # Operator evolution + injection + temporal encoding: [T, N, 2 * dynamic_hidden_dim]
        dynamic_embeddings = self.temporal_encoder(
            self.adj_matrices, static_sequence, node_feats
        )

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
