"""
Edge-imputation prediction script based on industry upstream/downstream dependencies.
[TENSOR STREAMING VARIANT — copy of get_imputation_fast.py, optimised for the 162M-pair scale]

Kept identical: config resolution, model loading, embedding cache semantics, threshold,
skip_existing semantics, output edge schema, all CLI switches.
Changed (see docs/implementation notes and TechnicalGuide):

  1. EmbeddingCache.predict_from_embeddings no longer runs a Python loop with one
     ``t_idx.item()`` GPU sync PER PAIR; the year -> time-step mapping is now a vectorised
     lookup table. At 162M candidate pairs that loop alone dominated the runtime.
  2. Candidate pairs are generated as GPU integer tensors in flat chunks instead of a
     per-block list of Python dicts (the largest industry block is 40.5M pairs -> ~12 GB).
  3. Existing-edge skipping no longer needs a per-pair dict lookup: the positions of the
     (rare) existing edges inside the flat index space are computed directly, O(#existing).
  4. Company lists of ALL industries come from ONE node scan instead of one full
     NodeByLabelScan per industry (the `'x' IN [i1, i2]` predicate cannot use any index).
  5. _query_existing_edges uses query parameters instead of interpolating thousands of ids
     into the Cypher string.
  6. Written edges MERGE on {year, source, model} and then SET the probability, so re-running
     with a slightly different score OVERWRITES instead of creating a parallel edge
     (the old 4-key MERGE created a new edge for every probability change).
  7. New performance.pair_chunk_size knob (default 1_000_000) plus the existing batch_size:
     pairs are streamed in chunks, each scored in batch_size slices.

Usage:
    python Prediction/get_imputation_tensor.py                     # uses model.config_name in the YAML
    python Prediction/get_imputation_tensor.py --config egcn        # CLI overrides the YAML
    python Prediction/get_imputation_tensor.py --checkpoint path/to/model.pth

Main parameters:
    --num-writers N    number of writer threads (1 recommended, avoids Neo4j deadlock)
    --flush-size N     number of edges accumulated before a batch write (default 500)
    --query-threads N  number of Neo4j query threads
    --min-degree N     minimum node degree used to build the graph (data.min_degree, 0 = every edge)
    --exclude-low-degree-nodes / --keep-low-degree-nodes
                       drop / keep nodes with degree < min_degree in the candidate lists
                       (data.exclude_low_degree_nodes)
"""

import sys
import os
import argparse
import json
import time
import logging
import importlib
import inspect
import itertools
import queue
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm import tqdm

# Compatibility: this environment ships numpy < 2, but checkpoints pickled by numpy >= 2
# reference `numpy._core.*` (numpy.core.* was renamed there). Alias the old package so those
# files keep loading instead of failing with "No module named 'numpy._core'".
# Pure fallback: on numpy >= 2 nothing is touched. Verified on numpy 1.26 + an egcn checkpoint.
if not hasattr(np, "_core"):
    import numpy.core as _numpy_core
    sys.modules.setdefault("numpy._core", _numpy_core)
    for _mod_name in list(sys.modules):
        if _mod_name.startswith("numpy.core"):
            sys.modules.setdefault(
                "numpy._core" + _mod_name[len("numpy.core"):], sys.modules[_mod_name]
            )

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils import deep_merge, import_attr, resolve_auto_kwargs, resolve_device

# Suppress Neo4j logs
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


# Config loading

IMPUTATION_YAML = os.path.join(ROOT_DIR, 'Prediction', 'imputation_common.yaml')
DEFAULT_MODEL_CONFIG = 'gatgru_vec'


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _imputation_yaml() -> dict:
    """Prediction/imputation_common.yaml (missing file -> defaults + a warning)"""
    if os.path.exists(IMPUTATION_YAML):
        return _load_yaml(IMPUTATION_YAML)
    print(f"[WARN] Imputation config file not found: {IMPUTATION_YAML}, using default parameters")
    return {}


def resolve_model_config_name(cli_config: Optional[str] = None) -> str:
    """Which Models/configs/<name>.yaml to load: CLI --config > imputation_common.yaml > default"""
    if cli_config:
        return cli_config
    from_yaml = (_imputation_yaml().get('model', {}) or {}).get('config_name', '')
    return from_yaml or DEFAULT_MODEL_CONFIG


def load_imputation_config(cli_config: Optional[str] = None) -> dict:
    """Load the full config: common_config.yaml -> model_config.yaml -> imputation_common.yaml.

    The model config name comes from the CLI when given, otherwise from
    imputation_common.yaml (model.config_name), so switching backbone needs no CLI flag.
    """
    model_config_name = resolve_model_config_name(cli_config)

    training_dir = os.path.join(ROOT_DIR, 'Training')
    common_path = os.path.join(training_dir, 'common_config.yaml')
    cfg = _load_yaml(common_path)

    model_config_dir = os.path.join(ROOT_DIR, 'Models', 'configs')
    yaml_path = os.path.join(model_config_dir, f"{model_config_name}.yaml")
    if os.path.exists(yaml_path):
        cfg = deep_merge(cfg, _load_yaml(yaml_path))
    elif os.path.exists(model_config_name):
        cfg = deep_merge(cfg, _load_yaml(model_config_name))
    else:
        raise FileNotFoundError(
            f"Model config not found: {model_config_name} "
            f"(looked in {model_config_dir})"
        )

    cfg = deep_merge(cfg, _imputation_yaml())

    # The resolved name is the single source of truth downstream (checkpoint search, per-model
    # threshold, embedding-cache fingerprint, output.model_name fallback).
    cfg.setdefault('model', {})['config_name'] = model_config_name
    cfg['model_config_name'] = model_config_name
    return cfg


# Embedding precomputation / cache

class EmbeddingCache:
    """Dynamic embedding precomputation cache manager (same as get_imputation.py)"""

    def __init__(self, model: nn.Module, year_to_idx: dict, num_timesteps: int,
                 meta: Optional[dict] = None):
        self.model = model
        self.year_to_idx = year_to_idx
        self.num_timesteps = num_timesteps
        self.dynamic_hidden_dim = None
        self._cached = None
        # Identity of the run the cached embeddings belong to (checkpoint / config / graph size).
        # A cached file whose meta differs is dropped and recomputed, so switching backbone or
        # checkpoint can never silently reuse another model's embeddings.
        self.meta = dict(meta or {})
        # Cached (raw year -> time-step index) lookup table; see year_position_lut().
        self._year_lut: Optional[torch.Tensor] = None
        self._year_lut_device: Optional[str] = None

    def year_position_lut(self, device: torch.device) -> torch.Tensor:
        """Raw-year -> time-step-index lookup tensor (-1 = year has no time step).

        Indexing this tensor replaces the old per-row ``for t_idx in time_indices:
        year_to_idx.get(int(t_idx.item()))`` Python loop, which cost one GPU->CPU
        synchronisation per candidate pair (162M pairs -> the dominant runtime share).
        """
        key = str(device)
        if self._year_lut is None or self._year_lut_device != key:
            size = int(max(int(y) for y in self.year_to_idx)) + 2 if self.year_to_idx else 2
            lut = torch.full((size,), -1, dtype=torch.long)
            for year, idx in self.year_to_idx.items():
                lut[int(year)] = int(idx)
            self._year_lut = lut.to(device)
            self._year_lut_device = key
        return self._year_lut

    def year_positions(self, time_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Vectorised raw-year -> time-step index (-1 where the year has no time step)."""
        lut = self.year_position_lut(device)
        in_range = (time_indices >= 0) & (time_indices < lut.numel())
        safe = torch.where(in_range, time_indices, torch.zeros_like(time_indices))
        positions = lut[safe]
        return torch.where(in_range, positions, torch.full_like(positions, -1))

    def is_compatible(self) -> bool:
        if not (hasattr(self.model, 'static_encoder') and
                hasattr(self.model, 'temporal_encoder') and
                hasattr(self.model, 'edge_predictor')):
            return False
        # Fusion backbones (FiLM / temporal injection) feed extra tensors into the temporal encoder,
        # which this cache path cannot supply -> fall back to the per-batch forward.
        try:
            params = inspect.signature(self.model.temporal_encoder.forward).parameters
        except (TypeError, ValueError):
            return True
        required = [p for p in params.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    and p.default is p.empty]
        return len(required) <= 1

    def precompute_embeddings(self, force: bool = False,
                               save_path: str = "",
                               device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not force and save_path and os.path.exists(save_path):
            loaded = self.load(save_path)
            if loaded is not None:
                return loaded
        try:
            embeddings = self.compute_embeddings(device)
            if save_path:
                self.save(save_path)
            return embeddings
        except Exception as e:
            print(f"[EmbeddingCache] Embedding computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compute_embeddings(self, device: torch.device) -> torch.Tensor:
        model = self.model
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'static_encoder') and hasattr(model, 'raw_node_feats'):
                encoder = model.static_encoder
                if hasattr(model, 'adj_matrices'):
                    static_sequence = encoder(model.adj_matrices, model.raw_node_feats)
                elif hasattr(model, 'edge_index_list'):
                    static_sequence = encoder(model.edge_index_list, model.raw_node_feats)
                else:
                    static_sequence = encoder(model.raw_node_feats)
            elif hasattr(model, 'subgraphs') and hasattr(model, 'static_encoder'):
                time_embeddings = []
                for subgraph in model.subgraphs:
                    emb = model.static_encoder(subgraph)
                    time_embeddings.append(emb.unsqueeze(0))
                static_sequence = torch.cat(time_embeddings, dim=0)
            else:
                raise RuntimeError(
                    f"Model {type(model).__name__} is missing the required attributes, "
                    f"cannot precompute embeddings. Use --no-cache to disable the embedding cache."
                )
            gru_input = static_sequence.transpose(0, 1)
            dynamic_sequence, _ = model.temporal_encoder(gru_input)
            embeddings = dynamic_sequence.transpose(0, 1)
        self.dynamic_hidden_dim = embeddings.size(-1) // 2
        self._cached = embeddings
        return embeddings

    def predict_from_embeddings(
        self, node_pairs: torch.Tensor, time_indices: torch.Tensor,
        dynamic_embeddings: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        # Vectorised year -> time-step mapping (was: one .item() GPU sync per pair).
        time_positions = self.year_positions(time_indices, device)
        valid_mask = time_positions >= 0
        if not valid_mask.all():
            if not valid_mask.any():
                return torch.tensor([], device=device)
            time_positions = time_positions[valid_mask]
            node_pairs = node_pairs[valid_mask]
        u_indices = node_pairs[:, 0].long()
        v_indices = node_pairs[:, 1].long()
        emb_u = dynamic_embeddings[time_positions, u_indices]
        emb_v = dynamic_embeddings[time_positions, v_indices]
        time_var = (time_positions.float() / self.num_timesteps).unsqueeze(1)
        pair_features = torch.cat([emb_u, emb_v, time_var], dim=-1)
        predictions = self.model.edge_predictor(pair_features).squeeze(-1)
        return predictions

    def save(self, path: str):
        if self._cached is not None:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save({
                'embeddings': self._cached.cpu(),
                'year_to_idx': self.year_to_idx,
                'num_timesteps': self.num_timesteps,
                'dynamic_hidden_dim': self.dynamic_hidden_dim,
                'meta': self.meta,
            }, path)
            print(f"[EmbeddingCache] Embeddings saved to: {path}")

    def load(self, path: str) -> Optional[torch.Tensor]:
        if not os.path.exists(path):
            return None
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            cached_meta = data.get('meta')
            if self.meta and cached_meta is not None and cached_meta != self.meta:
                print(f"[EmbeddingCache] Cache is stale ({cached_meta} != {self.meta}), recomputing")
                return None
            self._cached = data['embeddings']
            self.dynamic_hidden_dim = data.get('dynamic_hidden_dim')
            print(f"[EmbeddingCache] Embeddings loaded: {path}")
            print(f"  shape: {self._cached.shape}")
            return self._cached
        except Exception as e:
            print(f"[WARN] Failed to load embedding cache ({path}): {e}")
            return None


# Model loading

def load_model(cfg: dict, device: torch.device) -> Tuple[nn.Module, dict, Any, Any]:
    model_cfg = cfg['model']
    ds_cfg = cfg['dataset']
    impu_data_cfg = cfg.get('data', {})

    print("\n[1/4] Loading full dataset...")
    # data.min_degree (Prediction/imputation_common.yaml) drives the inference graph. 0 (default)
    # keeps every edge - the historical behaviour. A positive value applies the same EDGE filter as
    # training (both endpoints must reach the threshold), so the dynamic embeddings are computed on
    # a sparser graph. Node ids / num_nodes never change (they come from the embedding table).
    min_degree = int(impu_data_cfg.get('min_degree', 0) or 0)
    print(f"  min_degree: {min_degree} (0 = keep every edge)")
    DataModule = importlib.import_module(ds_cfg['module'])
    CompanySupplyDataset = getattr(DataModule, 'CompanySupplyDataset')
    build_static_graph = getattr(DataModule, 'build_static_graph')

    year_range = impu_data_cfg.get('year_range', list(range(2013, 2026)))
    cls_params = set(inspect.signature(CompanySupplyDataset.__init__).parameters.keys())

    full_dataset_kwargs = {
        'negative_ratio': 0,
        'embedding_name': impu_data_cfg.get('embedding_name', 'embedding'),
        'toy_mode': cfg.get('toy_mode', False),
        'min_degree': min_degree,
        'source_filter': impu_data_cfg.get('source_filter', 'semi'),
    }
    extra_candidates = {
        'filter_factset_neg': ds_cfg.get('filter_factset_neg', False),
        'intra_industry_neg': ds_cfg.get('intra_industry_neg', True),
        'use_pred_neg': ds_cfg.get('use_pred_neg', True),
        'use_attr_onehot': ds_cfg.get('use_attr_onehot', False),
        'onehot_attrs': ds_cfg.get('onehot_attrs', None),
        # Degree channel: must match the layout the checkpoint was trained on (d -> d + 1)
        'use_attr_degree': ds_cfg.get('use_attr_degree', False),
        'degree_property': ds_cfg.get('degree_property', 'degree'),
    }
    for k, v in extra_candidates.items():
        if k in cls_params:
            full_dataset_kwargs[k] = v
    full_dataset_kwargs = {k: v for k, v in full_dataset_kwargs.items() if k in cls_params}

    full_dataset = CompanySupplyDataset(**full_dataset_kwargs)

    print("\n[2/4] Building full graph data...")
    dynamic_data = build_static_graph(
        full_set=full_dataset, edge_info_dataset=full_dataset
    )
    dynamic_data = dynamic_data.to(device)
    print(f"  nodes: {dynamic_data.num_nodes}, edges: {dynamic_data.edge_index.shape[1]}")

    print("\n[3/4] Initializing model...")
    ModelClass = import_attr(model_cfg['module'], model_cfg['class'])
    print(f"  Model config: {cfg.get('model_config_name', 'N/A')} ({cfg.get('name', 'N/A')})")
    print(f"  Model class:  {model_cfg['module']}.{model_cfg['class']}")

    pinned_time_steps = impu_data_cfg.get('time_steps') or []
    if pinned_time_steps:
        # data.time_steps must reproduce the axis the checkpoint was trained on; keep the inference
        # default otherwise (year positions are the address into the temporal embeddings).
        time_steps = sorted(int(y) for y in pinned_time_steps)
    else:
        time_steps = sorted(dynamic_data.edge_time.unique().tolist())
        for y in year_range:
            if y not in time_steps:
                time_steps.append(y)
        time_steps = sorted(time_steps)
    print(f"  Time steps ({len(time_steps)}): {[int(t) for t in time_steps]}")

    auto_context = {
        'num_features': dynamic_data.x.size(1),
        'num_nodes': dynamic_data.num_nodes,
        'time_steps': time_steps,
        'hidden_dims': dynamic_data.x.size(1),
        'device': device,
    }
    model_kwargs = resolve_auto_kwargs(model_cfg.get('kwargs', {}), auto_context)

    init_params = inspect.signature(ModelClass.__init__).parameters
    if 'dynamic_data' in init_params:
        model = ModelClass(dynamic_data=dynamic_data, **model_kwargs).to(device)
    else:
        model = ModelClass(**model_kwargs).to(device)

    print("\n[4/4] Loading model weights...")
    check_prediction_interface(model)

    impu_model_cfg = cfg.get('model', {})
    checkpoint_path = impu_model_cfg.get('checkpoint', '') or _find_checkpoint(cfg, model_cfg)
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT_DIR, checkpoint_path)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Set model.checkpoint (or model.run_dir) in Prediction/imputation_common.yaml, "
            f"or pass --checkpoint"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    filtered_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            skipped.append(k)
    missing = [k for k in model_state if k not in filtered_dict]
    if skipped:
        print(f"  [WARN] Skipped {len(skipped)} checkpoint keys not matching the architecture "
              f"(e.g. {skipped[:3]})")
    if missing:
        print(f"  [WARN] {len(missing)} model parameters left at random init "
              f"(e.g. {missing[:3]})")
    if not filtered_dict:
        raise RuntimeError(
            f"No checkpoint parameter matched {model_cfg['module']}.{model_cfg['class']}: "
            f"the checkpoint was almost certainly trained on another backbone or feature width. "
            f"Check model.config_name / model.checkpoint in Prediction/imputation_common.yaml.\n"
            f"  checkpoint: {checkpoint_path}"
        )
    strict = bool(impu_model_cfg.get('strict', False))
    if strict and (skipped or missing):
        raise RuntimeError(
            f"model.strict=true but checkpoint and architecture disagree "
            f"(skipped={len(skipped)}, missing={len(missing)}).\n"
            f"  checkpoint: {checkpoint_path}\n"
            f"  model:      {model_cfg['module']}.{model_cfg['class']}"
        )

    model.load_state_dict(filtered_dict, strict=False)
    model.eval()
    print(f"  Model weights loaded: {checkpoint_path}")
    print(f"  Matched {len(filtered_dict)}/{len(model_state)} parameter tensors")
    for key in ('best_f1', 'best_auc', 'epoch'):
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")

    year_to_idx = {int(year): i for i, year in enumerate(time_steps)}
    model_info = {
        'year_to_idx': year_to_idx,
        'time_steps': time_steps,
        'num_nodes': dynamic_data.num_nodes,
        'num_features': int(dynamic_data.x.size(1)),
        'min_degree': min_degree,
        'checkpoint_path': checkpoint_path,
    }
    return model, model_info, full_dataset, dynamic_data


def compute_node_degrees(full_dataset) -> Dict[int, int]:
    """Degree per PyG node index, counted on the UNFILTERED positive edges.

    Same definition as the training filter (``Data/graph_dataset.py::_get_positive_samples``):
    degree = how often a node appears as an endpoint of a (src, tgt, year) triple. The count has to
    be taken BEFORE the min_degree edge filter, otherwise a node can drop below the threshold simply
    because its own edges were removed, and the exclusion list would snowball.

    Only called when data.exclude_low_degree_nodes asks for it, so the extra query is skipped
    otherwise. Nodes that never appear are absent from the result (degree 0).
    """
    emb_name = getattr(full_dataset, 'embedding_name', 'embedding')
    source_filter = getattr(full_dataset, 'source_filter', None)
    source_clause = f" AND r.source = '{source_filter}'" if source_filter else ""
    query = f"""
        MATCH (c1:EntityObj)-[r:SupplyProductTo]->(c2:EntityObj)
        WHERE c1.{emb_name} IS NOT NULL AND c2.{emb_name} IS NOT NULL
        AND r.year IS NOT NULL AND r.year >= 2013 AND r.year <= 2025{source_clause}
        RETURN id(c1) as source_id, id(c2) as target_id
    """
    node_mapping = full_dataset.node_mapping
    degree: Dict[int, int] = {}
    for record in full_dataset.neo4j_host.execute_query(query):
        src = node_mapping.get(record['source_id'])
        tgt = node_mapping.get(record['target_id'])
        if src is not None:
            degree[src] = degree.get(src, 0) + 1
        if tgt is not None:
            degree[tgt] = degree.get(tgt, 0) + 1
    print(f"  [min_degree] endpoint hits: {sum(degree.values()):,}; "
          f"nodes with degree > 0: {len(degree):,}/{len(node_mapping):,}")
    return degree


def check_prediction_interface(model: nn.Module) -> None:
    """This script scores a batch as model(node_pairs, time_indices).

    Temp-SEAL is the one backbone that does not fit: it needs k-hop subgraphs
    (forward(data, link_indices, current_times)) and has no whole-graph scoring entry point,
    so it must be rejected instead of being silently fed the wrong arguments.
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return
    positional = [p for p in params.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in positional if p.default is p.empty]
    if len(positional) < 2 or len(required) > 2:
        raise RuntimeError(
            f"{type(model).__name__}.forward expects "
            f"{[p.name for p in inspect.signature(model.forward).parameters.values()]}, this script "
            f"can only call forward(node_pairs, time_indices) (whole-graph scoring). "
            f"Backbones that need per-link subgraphs (e.g. Temp-SEAL) are not supported here."
        )


def _newest_checkpoint(root: str, ckpt_name: str) -> Optional[str]:
    """Newest <ckpt_name> at or below root (None when there is none)"""
    if not root or not os.path.isdir(root):
        return None
    direct = os.path.join(root, ckpt_name)
    if os.path.exists(direct):
        return direct
    candidates = []
    for dirpath, _dirnames, files in os.walk(root):
        if ckpt_name in files:
            p = os.path.join(dirpath, ckpt_name)
            candidates.append((os.path.getmtime(p), p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _find_checkpoint(cfg: dict, model_cfg: dict) -> Optional[str]:
    """Resolve the weights: model.checkpoint > model.run_dir > results/<output_subdir>/**"""
    impu_model_cfg = cfg.get('model', {}) or {}
    explicit = impu_model_cfg.get('checkpoint', '') or ''
    if explicit:
        return explicit

    ckpt_name = impu_model_cfg.get('checkpoint_name', 'best_model.pth')
    run_dir = impu_model_cfg.get('run_dir', '') or ''
    if run_dir:
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(ROOT_DIR, 'results', run_dir)
        found = _newest_checkpoint(run_dir, ckpt_name)
        if found:
            return found
        print(f"[WARN] No {ckpt_name} below model.run_dir: {run_dir}")

    output_subdir = cfg.get('output_subdir', '')
    if output_subdir:
        return _newest_checkpoint(os.path.join(ROOT_DIR, 'results', output_subdir), ckpt_name)
    return None


# Batch Neo4j writer (background threads, producer-consumer pattern)

class Neo4jBatchWriter:
    """Batch Neo4j write manager

    Uses background threads + a queue to implement a producer-consumer pattern:
    - Producer (GPU prediction main thread) pushes threshold-passing edges into the queue
    - Consumer (background thread) accumulates up to flush_size, then batch-writes to Neo4j via UNWIND

    Written edge properties: source='imputation', model=<model name>.
    """

    # NOTE (changed vs get_imputation_fast.py): `probability` is no longer part of the MERGE key.
    # It used to be, so re-running with a score that differs at the 4th decimal created a PARALLEL
    # edge instead of updating the existing one. The identity of an imputed edge is
    # (src, tgt, year, source, model); the score is then written with SET.
    _BATCH_WRITE_QUERY_TEMPLATE = """
        UNWIND $batch AS row
        MATCH (u:%(label)s), (v:%(label)s)
        WHERE id(u) = row.src AND id(v) = row.tgt
        MERGE (u)-[r:%(relation)s {
            year: row.year,
            source: '%(source)s',
            model: '%(model_name)s'
        }]->(v)
        SET r.probability = row.prob
        RETURN count(r) as written
    """

    def __init__(
        self,
        label_name: str,
        relation_name: str,
        output_source: str,
        output_model: str,
        flush_size: int = 5000,
        num_workers: int = 2,
    ):
        """
        Args:
            label_name: Neo4j node label
            relation_name: edge relation type
            output_source: source property value of the output edge
            output_model: model property value of the output edge
            flush_size: number of edges accumulated before a batch write
            num_workers: number of writer threads (each thread connects to Neo4j independently)
        """
        self.label_name = label_name
        self.relation_name = relation_name
        self.output_source = output_source
        self.output_model = output_model
        self.flush_size = flush_size
        self.num_workers = min(num_workers, 8)  # cap at 8 to avoid too many connections
        # NOTE: with num_workers > 1, multiple writers concurrently MERGE on overlapping
        # nodes will trigger Neo4j DeadlockDetected (Forseti lock conflict); num_workers=1
        # is recommended to avoid deadlocks.

        # Thread-safe queue (larger buffer: with batch MERGE, the writer is the bottleneck, so the
        # producer must not be throttled every 100 rows)
        self._queue: queue.Queue = queue.Queue(maxsize=20000)
        self._workers: List[threading.Thread] = []
        self._running = False
        self._total_written = 0
        self._lock = threading.Lock()
        self._flush_count = 0

        self._query = self._BATCH_WRITE_QUERY_TEMPLATE % {
            'label': self.label_name,
            'relation': self.relation_name,
            'source': self.output_source,
            'model_name': self.output_model,
        }

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    def start(self):
        """Start the background writer threads"""
        if self._running:
            return
        self._running = True
        from Data.neo4j_SPLC import Neo4jClient

        def _worker(worker_id: int):
            """Each worker connects to Neo4j independently to avoid connection contention"""
            client = Neo4jClient()
            buffer: List[Dict] = []
            while self._running:
                try:
                    # Blocking wait, 1 second timeout to check the _running flag
                    item = self._queue.get(timeout=1.0)
                    buffer.append(item)
                    # Flush immediately once flush_size is reached
                    if len(buffer) >= self.flush_size:
                        self._flush_buffer(client, buffer)
                        buffer = []
                except queue.Empty:
                    # Timed out with no data, check for leftover data to write
                    if buffer:
                        self._flush_buffer(client, buffer)
                        buffer = []
            # Flush leftover data before exiting
            if buffer:
                try:
                    self._flush_buffer(client, buffer)
                except Exception as e:
                    print(f"\n[Writer-{worker_id}] Flush before exit failed: {e}")

        for i in range(self.num_workers):
            t = threading.Thread(target=_worker, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)

    def _flush_buffer(self, client, batch: List[Dict], max_retries: int = 3):
        """Batch-write the buffer to Neo4j (single UNWIND statement), with deadlock retries"""
        if not batch:
            return

        last_error = None
        for attempt in range(max_retries):
            try:
                params = {"batch": batch}
                client.execute_query(self._query, parameters=params)
                with self._lock:
                    self._total_written += len(batch)
                    self._flush_count += 1
                return  # success, return directly
            except Exception as e:
                last_error = e
                err_msg = str(e)
                is_deadlock = 'DeadlockDetected' in err_msg or 'deadlock' in err_msg.lower()
                if is_deadlock and attempt < max_retries - 1:
                    wait = (2.0 ** attempt) + random.uniform(0, 1)
                    print(f"\n[Writer] Deadlock detected, retry {attempt+1} (waiting {wait:.1f}s)...")
                    time.sleep(wait)
                else:
                    break

        # All retries failed, fall back to single-row writes
        print(f"\n[Writer] Batch write failed ({len(batch)} rows): {last_error}, falling back to single-row writes...")
        for item in batch:
            try:
                self._write_single(client, item)
            except Exception as e2:
                err2 = str(e2)
                if 'DeadlockDetected' in err2 or 'deadlock' in err2.lower():
                    # Single-row deadlock: brief wait, then retry once
                    time.sleep(random.uniform(0.5, 1.5))
                    try:
                        self._write_single(client, item)
                    except Exception as e3:
                        print(f"  [Writer] Single-row write deadlock retry still failed: {e3}")
                else:
                    print(f"  [Writer] Single-row write failed: {e2}")

    def _write_single(self, client, item: Dict):
        """Fallback: single-row write"""
        query = f"""
            MATCH (u:{self.label_name}), (v:{self.label_name})
            WHERE id(u) = {item['src']} AND id(v) = {item['tgt']}
            MERGE (u)-[r:{self.relation_name} {{
                year: {item['year']},
                source: '{self.output_source}',
                model: '{self.output_model}'
            }}]->(v)
            SET r.probability = {item['prob']:.4f}
        """
        client.execute_query(query)
        with self._lock:
            self._total_written += 1

    def enqueue(self, src_neo4j: int, tgt_neo4j: int, year: int, prob: float):
        """Push a threshold-passing edge into the write queue (non-blocking)"""
        self._queue.put({
            'src': src_neo4j,
            'tgt': tgt_neo4j,
            'year': year,
            'prob': round(prob, 4),
        })

    def stop(self):
        """Stop background threads and wait for the queue to drain.

        The queue is drained BEFORE _running is cleared: the worker checks the flag only at the
        top of its loop, so stopping first would silently drop everything still buffered in the
        queue (up to maxsize rows now that the buffer is much larger than the old 100).
        """
        pending = self._queue.qsize()
        if pending:
            print(f"\n[Writer] Waiting for write queue to drain ({pending} pending)...")
        deadline = time.time() + 1800
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.2)
        self._running = False
        # The worker does one last 1 s wait, flushes its own remainder, then exits and flushes
        # again, so give it a generous join window (UNWIND batches are now much larger).
        for t in self._workers:
            t.join(timeout=600)
        print(f"[Writer] Background writing complete: {self.total_written:,} rows total, "
              f"{self._flush_count} batch writes")


class NullWriter:
    """Counting stand-in used by --dry-run: same interface as Neo4jBatchWriter, no Neo4j calls.

    Lets you measure scoring throughput and tune batch_size / pair_chunk_size / flush_size
    without touching the graph.
    """

    def __init__(self, *args, **kwargs):
        self.total_written = 0
        self._flush_count = 0
        self._queue = queue.Queue()

    def start(self):
        print("[INFO] dry-run active: NOTHING is written to Neo4j")

    def enqueue(self, src_neo4j: int, tgt_neo4j: int, year: int, prob: float):
        self.total_written += 1

    def stop(self):
        print(f"[Writer] dry-run: {self.total_written:,} edges would have been written")


# High-performance predictor

class ImputationPredictorFast:
    """Edge-imputation predictor

    Candidate edges are generated lazily via itertools.product to avoid the memory
    overhead of a large list; GPU prediction and Neo4j batch writes run in parallel
    through background threads.
    """

    def __init__(
        self,
        model: nn.Module,
        full_dataset,
        cfg: dict,
        model_info: dict,
        device: torch.device,
        embedding_cache: Optional[EmbeddingCache] = None,
    ):
        self.model = model
        self.full_dataset = full_dataset
        self.cfg = cfg
        self.device = device
        self.cache = embedding_cache
        self.year_to_idx = model_info['year_to_idx']
        self.time_steps = model_info['time_steps']

        impu_cfg = cfg.get('prediction', {})
        data_cfg = cfg.get('data', {})
        output_cfg = cfg.get('output', {})
        perf_cfg = cfg.get('performance', {})

        self.threshold = impu_cfg.get('threshold', 0.5)
        self.batch_size = impu_cfg.get('batch_size', 512)
        self.test_mode = impu_cfg.get('test_mode', False)
        self.test_industry = impu_cfg.get('test_industry', 'example_industry')
        self.skip_existing = impu_cfg.get('skip_existing', True)
        self.dry_run = bool(impu_cfg.get('dry_run', False))
        self.year_range = data_cfg.get('year_range', list(range(2013, 2026)))
        # min_degree was applied when the graph was built (see load_model); it is re-read here only
        # so the predictor can report it and, when asked, drop low-degree nodes from the candidates.
        self.min_degree = int(data_cfg.get('min_degree', 0) or 0)
        self.exclude_low_degree_nodes = bool(data_cfg.get('exclude_low_degree_nodes', False))

        self.label_name = data_cfg.get('label_name', 'EntityObj')
        self.embedding_name = data_cfg.get('embedding_name', 'embedding')
        self.relation_name = output_cfg.get('relation_name', 'SupplyProductTo')
        self.output_source = output_cfg.get('source', 'imputation')
        self.output_model = (output_cfg.get('model_name', '')
                             or cfg.get('name', '')
                             or cfg.get('model_config_name', 'Unknown'))

        # Performance config
        self.num_writers = perf_cfg.get('num_writers', 1)
        # Default 5000 instead of 500: one UNWIND round trip per 500 rows means ~194k transactions
        # for a 97M-edge run, i.e. hours spent purely on round trips + locking overhead.
        self.flush_size = perf_cfg.get('flush_size', 5000)
        self.query_threads = perf_cfg.get('query_threads', 8)
        # How many candidate pairs are materialised at once on the device before being sliced into
        # batch_size-width chunks for the model. 1e6 x int64 x 3 tensors ~= 24 MB on the GPU.
        self.pair_chunk_size = int(perf_cfg.get('pair_chunk_size', 1_000_000) or 1_000_000)
        if self.flush_size < 2000:
            print(f"[WARN] performance.flush_size={self.flush_size} is low for this script "
                  f"(~1 UNWIND round trip per {self.flush_size} written edges); "
                  f"try 5000-20000 with --flush-size")

        # Embeddings
        self._dynamic_embeddings: Optional[torch.Tensor] = None
        self.reverse_mapping = full_dataset.reverse_node_mapping
        # raw year -> position inside the flat (up, down, year) index space helper
        self._year_step = [int(self.year_to_idx.get(int(y), -1)) for y in self.year_range]


    def precompute_embeddings(self, force: bool = False) -> Optional[torch.Tensor]:
        if self.cache is None or not self.cache.is_compatible():
            print("[INFO] Model does not support embedding precomputation, will use per-batch forward inference")
            if self.batch_size < 4096:
                # Measured on EGCN with 51,509 nodes on a GPU: the graph forward costs ~270 ms per
                # call almost independently of how many pairs it scores, so a 512-pair batch pays
                # it 64x more often than a 32768-pair batch (1.9k vs 64k pairs/s).
                print(f"[PERF] prediction.batch_size={self.batch_size} is very small for a model "
                      f"without an embedding cache: every batch re-runs the FULL graph forward. "
                      f"Raising it to 16384-32768 typically gives a 20-30x speed-up here "
                      f"(--batch-size); only keep it small for models that build per-pair "
                      f"subgraphs (e.g. SEAL).")
            return None

        impu_cfg = self.cfg.get('embedding_cache', {})
        save_path = impu_cfg.get('save_path', '') or os.path.join(
            ROOT_DIR, 'results', self.cfg.get('output_subdir', 'default'),
            'imputation_embeddings.pt'
        )
        force_recompute = force or impu_cfg.get('force_recompute', False)

        embeddings = self.cache.precompute_embeddings(
            force=force_recompute, save_path=save_path, device=self.device
        )
        if embeddings is not None:
            self._dynamic_embeddings = embeddings.to(self.device)
        return embeddings


    def load_industry_network(self) -> dict:
        indus_cfg = self.cfg.get('indus_network', {})
        network_path = indus_cfg.get('path', 'info/indus_network.json')
        full_path = os.path.join(ROOT_DIR, network_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Industry network file not found: {full_path}")
        with open(full_path, 'r', encoding='utf-8') as f:
            indus_network = json.load(f)
        print(f"\n[Industry network] Loaded dependencies for {len(indus_network)} industries")
        if self.test_mode:
            if self.test_industry in indus_network:
                indus_network = {self.test_industry: indus_network[self.test_industry]}
                print(f"[Test mode] Processing industry only: '{self.test_industry}'")
            else:
                available = list(indus_network.keys())[:10]
                raise ValueError(f"Test industry '{self.test_industry}' not in industry network. Available: {available}")
        return indus_network


    def _get_industry_companies(self, industry: str) -> List[int]:
        """Get company node indices for the given industry (in-graph PyG indices)"""
        from Data.neo4j_SPLC import Neo4jClient
        client = Neo4jClient()
        result = client.execute_query(f"""
            MATCH (c:{self.label_name})
            WHERE '{industry}' in [c.industry_1st, c.industry_2nd]
              AND c.{self.embedding_name} IS NOT NULL
            RETURN id(c) as company_id
        """)
        if not result:
            return []
        neo4j_ids = [r['company_id'] for r in result]
        node_mapping = self.full_dataset.node_mapping
        company_ids = [
            node_mapping[nid] for nid in neo4j_ids if nid in node_mapping
        ]
        return company_ids

    def _get_all_industry_companies(self, indus_network: dict) -> Dict[str, List[int]]:
        """Fetch company lists for all industries with ONE node scan.

        The previous version issued one query per industry with
        ``WHERE '<industry>' IN [c.industry_1st, c.industry_2nd]``. That predicate is a list
        membership test on a computed list, so it is not index-backed: the planner falls back to
        ``NodeByLabelScan`` over every EntityObj node (measured 1.17 s per industry x tens of
        industries). One scan returning (id, industry_1st, industry_2nd) costs 4.34 s in total,
        and bucketing in Python also avoids re-doing the node_mapping lookup per industry.
        """
        from Data.neo4j_SPLC import Neo4jClient

        wanted = set()
        for downstream_ind, upstream_list in indus_network.items():
            wanted.add(downstream_ind)
            for item in upstream_list:
                if isinstance(item, list):
                    wanted.update(item)
                else:
                    wanted.add(item)

        print("\n[Preload] Fetching industry company info with a single node scan...")
        client = Neo4jClient()
        records = client.execute_query(f"""
            MATCH (c:{self.label_name})
            WHERE c.{self.embedding_name} IS NOT NULL
            RETURN id(c) as company_id, c.industry_1st as i1, c.industry_2nd as i2
        """)

        node_mapping = self.full_dataset.node_mapping
        result: Dict[str, List[int]] = {}
        for row in tqdm(records, desc="group companies", unit="node", dynamic_ncols=True):
            idx = node_mapping.get(row['company_id'])
            if idx is None:
                continue
            i1, i2 = row['i1'], row['i2']
            if i1 is not None and i1 in wanted:
                result.setdefault(i1, []).append(idx)
            if i2 is not None and i2 in wanted and i2 != i1:
                result.setdefault(i2, []).append(idx)

        missing = sorted(wanted - set(result))
        if missing:
            print(f"  [INFO] {len(missing)} industries have no company node and are skipped: "
                  f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
        print(f"  Fetched company lists for {len(result)} industries")
        total_companies = sum(len(v) for v in result.values())
        print(f"  {total_companies:,} company nodes in total")
        return result


    def _filter_low_degree_companies(
        self, industry_companies: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        """Drop nodes whose degree < data.min_degree from the imputation candidate lists.

        Degree uses the training definition (endpoint count over the UNFILTERED positives); a node
        missing from the degree map has degree 0 and is dropped as well. Keeping only nodes that were
        "visible enough" in the training graph avoids extrapolating the threshold to nodes the model
        never really saw.
        """
        degrees = compute_node_degrees(self.full_dataset)
        keep = {node for node, deg in degrees.items() if deg >= self.min_degree}

        filtered: Dict[str, List[int]] = {}
        total = 0
        removed = 0
        for industry, ids in industry_companies.items():
            total += len(ids)
            kept = [i for i in ids if i in keep]
            removed += len(ids) - len(kept)
            if kept:
                filtered[industry] = kept
        print(f"[min_degree] excluded {removed:,}/{total:,} company nodes with degree "
              f"< {self.min_degree}")
        print(f"  {len(keep):,} candidate nodes remain, {len(filtered)} industries still active")
        return filtered

    def _query_existing_edges(
        self, upstream_neo4j_ids: List[int],
        downstream_neo4j_ids: List[int]
    ) -> set:
        """Batch-query already-existing semi edges.

        The id lists are now passed as query parameters: interpolating several thousand ids into
        the Cypher string makes every one of these queries a unique string that must be parsed and
        planned from scratch (there are ~168 industry blocks).
        """
        from Data.neo4j_SPLC import Neo4jClient
        client = Neo4jClient()
        existing = set()
        result = client.execute_query(
            f"""
            MATCH (u:{self.label_name})-[r:{self.relation_name} {{source: 'semi'}}]->(v:{self.label_name})
            WHERE id(u) IN $up_ids
              AND id(v) IN $down_ids
              AND r.year IN $years
            RETURN id(u) as u_id, id(v) as v_id, r.year as year
            """,
            parameters={
                'up_ids': list(upstream_neo4j_ids),
                'down_ids': list(downstream_neo4j_ids),
                'years': list(self.year_range),
            },
        )
        if result:
            for record in result:
                existing.add((record['u_id'], record['v_id'], record['year']))
        return existing

    def _block_invalid_positions(
        self,
        upstream_ids: List[int],
        downstream_ids: List[int],
        up_pos: Dict[int, int],
        down_pos: Dict[int, int],
        existing_set: set,
        num_down: int,
        num_years: int,
    ) -> List[int]:
        """Positions inside the flat (up, down, year) index space that must NOT be scored.

        flat position of (up_i, down_j, year_k) = (i * num_down + j) * num_years + k.

        Existing edges are rare (1259 out of 236M in the largest block), so instead of testing
        every candidate pair against the existing set we invert the lookup: map each existing
        edge back to its position. Cost is O(#existing) instead of O(#candidates).
        Years that have no time step are added in the same way.
        """
        year_pos = {year: k for k, year in enumerate(self.year_range)}
        invalid = set()
        for u_nb, v_nb, year in existing_set:
            i = up_pos.get(u_nb)
            j = down_pos.get(v_nb)
            k = year_pos.get(year)
            if i is None or j is None or k is None:
                continue
            invalid.add((i * num_down + j) * num_years + k)

        for k, step in enumerate(self._year_step):
            if step < 0:  # year has no time step -> cannot be scored at all
                invalid.update(range(k, num_down * len(upstream_ids) * num_years, num_years))

        return sorted(invalid)


    def predict_batch(
        self, node_pairs: torch.Tensor, time_indices: torch.Tensor
    ) -> torch.Tensor:
        if self._dynamic_embeddings is not None and self.cache is not None:
            return self.cache.predict_from_embeddings(
                node_pairs, time_indices,
                self._dynamic_embeddings, self.device
            )
        else:
            with torch.no_grad():
                return self.model(node_pairs, time_indices)


    @staticmethod
    def _generate_candidate_pairs(upstream_ids, downstream_ids, year_range):
        """Lazily generate candidate pairs

        Yields: (src_idx, tgt_idx, year)

        NOTE: kept for backwards compatibility only; the scoring loop in run() no longer uses it,
        because materialising one Python tuple/dict per candidate pair costs ~12 GB of RAM on the
        largest industry block (40.5M pairs) and dominates the runtime at the 162M-pair scale.
        """
        for src, tgt, year in itertools.product(upstream_ids, downstream_ids, year_range):
            yield (src, tgt, year)


    def _flatten_upstream(self, upstream_list) -> List[str]:
        """Flatten the upstream industry list into a list of strings"""
        result = []
        for item in upstream_list:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def estimate_total_pairs(self, indus_network: dict,
                             industry_companies: Dict[str, List[int]]) -> Tuple[int, dict]:
        """Estimate the total number of prediction pairs"""
        print("\n" + "=" * 60)
        print("Counting candidate edges to predict")
        print("=" * 60)
        total_pairs = 0
        industry_stats = {}
        num_years = len(self.year_range)

        for downstream_ind, upstream_list in indus_network.items():
            d_ids = industry_companies.get(downstream_ind, [])
            if not d_ids:
                continue
            industry_stats[downstream_ind] = {
                'downstream_count': len(d_ids), 'upstream_pairs': []
            }
            for upstream_ind in self._flatten_upstream(upstream_list):
                u_ids = industry_companies.get(upstream_ind, [])
                if not u_ids:
                    continue
                pair_count = len(u_ids) * len(d_ids) * num_years
                total_pairs += pair_count
                industry_stats[downstream_ind]['upstream_pairs'].append({
                    'upstream_industry': upstream_ind,
                    'upstream_count': len(u_ids),
                    'pair_count': pair_count,
                })

        print(f"\nTotal candidate edges to predict: {total_pairs:,}")
        print(f"Number of downstream industries: {len(industry_stats)}")
        print(f"Year range: {min(self.year_range)}-{max(self.year_range)}")
        return total_pairs, industry_stats

    def run(self, indus_network: dict):
        """Run the full prediction and imputation pipeline (optimized version)"""
        print("\n" + "=" * 60)
        print("Starting prediction and imputation")
        print("=" * 60)
        print(f"Batch size: {self.batch_size}")
        print(f"Prediction threshold: {self.threshold}")
        print(f"Skip existing: {self.skip_existing}")
        print(f"Graph min_degree: {self.min_degree} (0 = every edge kept)")
        print(f"Exclude low-degree nodes: {self.exclude_low_degree_nodes}"
              f"{' (degree < ' + str(self.min_degree) + ')' if self.min_degree > 0 else ''}")
        print(f"Embedding precompute: {'enabled' if self._dynamic_embeddings is not None else 'disabled'}")
        print(f"Writer threads: {self.num_writers}")
        print(f"Batch write size: {self.flush_size}")
        print(f"Query threads: {self.query_threads}")
        print(f"Model batch size: {self.batch_size}")
        print(f"Pair streaming chunk: {self.pair_chunk_size:,}")

        reverse_mapping = self.full_dataset.reverse_node_mapping

        industry_companies = self._get_all_industry_companies(indus_network)
        if self.exclude_low_degree_nodes and self.min_degree > 0:
            industry_companies = self._filter_low_degree_companies(industry_companies)
        elif self.exclude_low_degree_nodes:
            print("[WARN] data.exclude_low_degree_nodes=true but data.min_degree <= 0: "
                  "nothing to exclude (set data.min_degree > 0)")

        total_estimate, _ = self.estimate_total_pairs(indus_network, industry_companies)

        writer_cls = NullWriter if self.dry_run else Neo4jBatchWriter
        writer = writer_cls(
            label_name=self.label_name,
            relation_name=self.relation_name,
            output_source=self.output_source,
            output_model=self.output_model,
            flush_size=self.flush_size,
            num_workers=self.num_writers,
        )
        writer.start()

        total_processed = 0
        skipped_count = 0
        saved_count = 0
        start_time = time.time()
        last_postfix = 0.0

        pbar = tqdm(total=total_estimate, desc="prediction progress", unit="pairs",
                     dynamic_ncols=True)

        for downstream_ind, upstream_list in indus_network.items():
            downstream_ids = industry_companies.get(downstream_ind, [])
            if not downstream_ids:
                continue

            for upstream_ind in self._flatten_upstream(upstream_list):
                upstream_ids = industry_companies.get(upstream_ind, [])
                if not upstream_ids:
                    continue

                # Nodes without a Neo4j id cannot be written back, so they are dropped here
                # (the old loop did the same check per candidate pair).
                up_ids, up_neo4j, down_ids, down_neo4j = [], [], [], []
                for uid in upstream_ids:
                    nb = reverse_mapping.get(uid)
                    if nb is not None:
                        up_ids.append(uid)
                        up_neo4j.append(nb)
                for did in downstream_ids:
                    nb = reverse_mapping.get(did)
                    if nb is not None:
                        down_ids.append(did)
                        down_neo4j.append(nb)
                upstream_ids, downstream_ids = up_ids, down_ids

                num_up, num_down = len(upstream_ids), len(downstream_ids)
                num_years = len(self.year_range)
                block_total = num_up * num_down * num_years
                if not upstream_ids or not downstream_ids:
                    pbar.update(block_total)
                    continue

                existing_set = set()
                if self.skip_existing:
                    existing_set = self._query_existing_edges(up_neo4j, down_neo4j)

                # Flat index space of this block: pos = (i * num_down + j) * num_years + k,
                # matching itertools.product(upstream_ids, downstream_ids, year_range) exactly.
                invalid_positions: List[int] = []
                if self.skip_existing and existing_set:
                    up_pos = {nb: i for i, nb in enumerate(up_neo4j)}
                    down_pos = {nb: j for j, nb in enumerate(down_neo4j)}
                    invalid_positions = self._block_invalid_positions(
                        upstream_ids, downstream_ids, up_pos, down_pos,
                        existing_set, num_down, num_years
                    )
                elif any(step < 0 for step in self._year_step):
                    invalid_positions = self._block_invalid_positions(
                        upstream_ids, downstream_ids, {}, {}, set(), num_down, num_years
                    )
                invalid_tensor = torch.tensor(
                    invalid_positions, dtype=torch.long, device=self.device
                ) if invalid_positions else None
                skipped_count += len(invalid_positions)

                # Device-side id tables for this block (built once, reused by every chunk)
                up_t = torch.as_tensor(upstream_ids, dtype=torch.long, device=self.device)
                down_t = torch.as_tensor(downstream_ids, dtype=torch.long, device=self.device)
                years_t = torch.as_tensor(list(self.year_range),
                                          dtype=torch.long, device=self.device)
                up_nb_t = torch.as_tensor(up_neo4j, dtype=torch.long, device=self.device)
                down_nb_t = torch.as_tensor(down_neo4j, dtype=torch.long, device=self.device)

                for chunk_start in range(0, block_total, self.pair_chunk_size):
                    chunk_end = min(chunk_start + self.pair_chunk_size, block_total)
                    flat = torch.arange(chunk_start, chunk_end,
                                        dtype=torch.long, device=self.device)

                    if invalid_tensor is not None:
                        local = invalid_tensor[
                            (invalid_tensor >= chunk_start) & (invalid_tensor < chunk_end)
                        ] - chunk_start
                        if local.numel():
                            keep = torch.ones(flat.numel(), dtype=torch.bool,
                                              device=self.device)
                            keep[local] = False
                            flat = flat[keep]

                    chunk_size = int(flat.numel())
                    if chunk_size == 0:
                        pbar.update(chunk_end - chunk_start)
                        continue

                    k = flat % num_years
                    base = flat // num_years
                    j = base % num_down
                    i = base // num_down

                    node_pairs = torch.stack([up_t[i], down_t[j]], dim=1)

                    # Slices of batch_size so GPU memory stays bounded (identical to the old loop)
                    for s in range(0, chunk_size, self.batch_size):
                        sl = slice(s, min(s + self.batch_size, chunk_size))
                        probs = self.predict_batch(node_pairs[sl], years_t[k[sl]])
                        if probs.numel() != (sl.stop - sl.start):
                            raise RuntimeError(
                                f"predict_batch returned {probs.numel()} scores for "
                                f"{sl.stop - sl.start} pairs - rows were dropped internally "
                                f"(a year has no time step); check data.year_range against the "
                                f"training time steps."
                            )

                        probs_np = probs.detach().cpu().numpy()
                        hit_pos = np.flatnonzero(probs_np > self.threshold)

                        if hit_pos.size:
                            hit_t = torch.as_tensor(hit_pos, dtype=torch.long,
                                                    device=self.device)
                            src_nb = up_nb_t[i[hit_t]].cpu().numpy()
                            tgt_nb = down_nb_t[j[hit_t]].cpu().numpy()
                            hit_year = years_t[k[hit_t]].cpu().numpy()
                            hit_prob = probs_np[hit_pos]
                            saved_count += int(hit_pos.size)
                            for src_id, tgt_id, year_v, prob in zip(
                                src_nb.tolist(), tgt_nb.tolist(),
                                hit_year.tolist(), hit_prob.tolist()
                            ):
                                writer.enqueue(src_id, tgt_id, year_v, prob)

                    total_processed += chunk_size
                    pbar.update(chunk_end - chunk_start)

                    # Progress update (throttled: the postfix formatting costs string work)
                    if time.time() - last_postfix > 2.0:
                        last_postfix = time.time()
                        elapsed = time.time() - start_time
                        speed = total_processed / elapsed if elapsed > 0 else 0.0
                        remaining = max(total_estimate - pbar.n, 0)
                        eta = remaining / speed if speed > 0 else 0
                        pbar.set_postfix({
                            'score>thr': saved_count,
                            'written': writer.total_written,
                            'speed': f'{speed:.0f}pairs/s',
                            'ETA': str(timedelta(seconds=int(eta))),
                            'queue': writer._queue.qsize(),
                        })

        pbar.close()

        writer.stop()

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Prediction and imputation complete")
        print("=" * 60)
        print(f"Total pairs predicted: {total_processed:,}")
        print(f"Skipped existing edges: {skipped_count:,}")
        print(f"Edges above threshold: {saved_count:,}")
        print(f"Actually written to Neo4j: {writer.total_written:,} ({writer._flush_count} batch writes)")
        if total_processed > 0:
            print(f"Hit rate: {saved_count/total_processed*100:.2f}%")
        print(f"Total elapsed: {timedelta(seconds=int(total_elapsed))}")
        print(f"Average speed: {total_processed/total_elapsed:.1f} pairs/s")
        print("=" * 60)


# Main entry

def main():
    parser = argparse.ArgumentParser(
        description='Temporal GNN edge imputation prediction [high-performance version] - batch writes + async pipeline + multi-threading'
    )
    parser.add_argument('--config', '-c', default=None,
                        help='model config name; overrides model.config_name in imputation_common.yaml')
    parser.add_argument('--checkpoint', '-p', default='',
                        help='model checkpoint path; overrides model.checkpoint in imputation_common.yaml')
    parser.add_argument('--device', '-d', default='auto',
                        help='device (auto/cuda/cpu); falls back to model.device in imputation_common.yaml')
    parser.add_argument('--indus-network', default='',
                        help='industry network JSON file path')
    parser.add_argument('--threshold', type=float, default=None,
                        help='prediction probability threshold')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='GPU batch size')
    parser.add_argument('--no-cache', action='store_true',
                        help='disable the embedding precompute cache')
    parser.add_argument('--force-recompute', action='store_true',
                        help='force recomputation of embeddings')
    parser.add_argument('--test', action='store_true',
                        help='enable test mode')
    parser.add_argument('--test-industry', default='',
                        help='industry name to process in test mode')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='do not skip existing edges')
    parser.add_argument('--min-degree', type=int, default=None,
                        help='minimum node degree used to build the inference graph '
                             '(overrides data.min_degree; 0 = keep every edge)')
    parser.add_argument('--exclude-low-degree-nodes', dest='exclude_low_degree_nodes',
                        action='store_true', default=None,
                        help='exclude nodes with degree < min_degree from the imputation candidates '
                             '(overrides data.exclude_low_degree_nodes)')
    parser.add_argument('--keep-low-degree-nodes', dest='exclude_low_degree_nodes',
                        action='store_false', default=None,
                        help='keep every embedded node as a candidate '
                             '(overrides data.exclude_low_degree_nodes)')
    parser.add_argument('--num-writers', type=int, default=None,
                        help='number of writer threads (default 4)')
    parser.add_argument('--flush-size', type=int, default=None,
                        help='number of edges accumulated before a batch write (default 500)')
    parser.add_argument('--query-threads', type=int, default=None,
                        help='number of Neo4j query threads (default 8)')
    parser.add_argument('--dry-run', action='store_true',
                        help='score everything but write nothing to Neo4j (tuning aid)')
    parser.add_argument('--pair-chunk-size', type=int, default=None,
                        help='how many candidate pairs are materialised on the device at once '
                             '(performance.pair_chunk_size, default 1000000; each chunk is then '
                             'scored in batch_size slices)')

    args = parser.parse_args()

    print("=" * 60)
    print("Temporal GNN edge imputation prediction [tensor streaming variant]")
    print("=" * 60)

    cfg = load_imputation_config(args.config)
    model_config_name = cfg['model_config_name']

    # Device: CLI > model.device in imputation_common.yaml > auto
    device_str = resolve_device(
        args.device if args.device != 'auto'
        else (cfg.get('model', {}) or {}).get('device', 'auto')
    )
    device = torch.device(device_str)

    # Command-line argument overrides
    if args.checkpoint:
        cfg.setdefault('model', {})['checkpoint'] = args.checkpoint
    if args.indus_network:
        cfg.setdefault('indus_network', {})['path'] = args.indus_network
    if args.threshold is not None:
        cfg.setdefault('prediction', {})['threshold'] = args.threshold
    if args.batch_size is not None:
        cfg.setdefault('prediction', {})['batch_size'] = args.batch_size
    if args.test:
        cfg.setdefault('prediction', {})['test_mode'] = True
    if args.test_industry:
        cfg.setdefault('prediction', {})['test_industry'] = args.test_industry
    if args.no_skip_existing:
        cfg.setdefault('prediction', {})['skip_existing'] = False
    if args.dry_run:
        cfg.setdefault('prediction', {})['dry_run'] = True
    if args.min_degree is not None:
        cfg.setdefault('data', {})['min_degree'] = args.min_degree
    if args.exclude_low_degree_nodes is not None:
        cfg.setdefault('data', {})['exclude_low_degree_nodes'] = args.exclude_low_degree_nodes
    if args.no_cache:
        cfg.setdefault('embedding_cache', {})['enabled'] = False
    if args.force_recompute:
        cfg.setdefault('embedding_cache', {})['force_recompute'] = True

    # Performance parameters
    perf_cfg = cfg.setdefault('performance', {})
    if args.num_writers is not None:
        perf_cfg['num_writers'] = args.num_writers
    if args.flush_size is not None:
        perf_cfg['flush_size'] = args.flush_size
    if args.query_threads is not None:
        perf_cfg['query_threads'] = args.query_threads
    if args.pair_chunk_size is not None:
        perf_cfg['pair_chunk_size'] = args.pair_chunk_size

    model_cfg = cfg['model']
    impu_pred_cfg = cfg.get('prediction', {})
    perf_cfg_final = cfg.get('performance', {})

    # Per-model threshold: score scales are not comparable across backbones, so an entry in
    # prediction.threshold_by_model wins; the CLI value (applied above) always wins over both.
    if args.threshold is None:
        by_model = impu_pred_cfg.get('threshold_by_model') or {}
        if model_config_name in by_model:
            impu_pred_cfg['threshold'] = float(by_model[model_config_name])
            threshold_source = f"prediction.threshold_by_model[{model_config_name}]"
        else:
            threshold_source = (f"prediction.threshold (no threshold_by_model entry for "
                                f"'{model_config_name}')")
            print(f"[WARN] Score scales are not comparable across backbones: fill "
                  f"prediction.threshold_by_model['{model_config_name}'] (Youden's J / F1-max from "
                  f"Prediction/find_threshold*.py) for this model before writing to Neo4j.")
    else:
        threshold_source = "--threshold"

    print(f"Model config: {model_config_name}"
          f"{' (from imputation_common.yaml)' if not args.config else ' (from --config)'}")
    print(f"Model class:  {model_cfg.get('class', 'N/A')}")
    print(f"Checkpoint:   {model_cfg.get('checkpoint') or model_cfg.get('run_dir') or 'auto-search'}")
    print(f"Device:       {device}")
    print(f"Threshold:    {impu_pred_cfg.get('threshold', 0.5)}  <- {threshold_source}")
    print(f"Writers:      {perf_cfg_final.get('num_writers', 1)}")
    print(f"Batch write:  {perf_cfg_final.get('flush_size', 5000)} rows/write")
    print(f"Pair chunk:   {perf_cfg_final.get('pair_chunk_size', 1_000_000):,} pairs "
          f"(scored in slices of {impu_pred_cfg.get('batch_size', 512)})")
    if impu_pred_cfg.get('test_mode'):
        print(f"Test mode: processing only '{impu_pred_cfg.get('test_industry', 'example_industry')}'")
    impu_data_cfg = cfg.get('data', {})
    min_degree = int(impu_data_cfg.get('min_degree', 0) or 0)
    print(f"Graph min_degree: {min_degree}"
          f"{' (training-like edge filter)' if min_degree > 0 else ' (keep every edge)'}")
    exclude_low = bool(impu_data_cfg.get('exclude_low_degree_nodes', False))
    if min_degree > 0:
        print(f"Exclude low-degree candidates: {exclude_low}"
              f"{' (degree < ' + str(min_degree) + ')' if exclude_low else ''}")
    elif exclude_low:
        print("[WARN] data.exclude_low_degree_nodes=true but data.min_degree <= 0: ignored, "
              "no node is excluded")

    # 1. Load model and data
    model, model_info, full_dataset, dynamic_data = load_model(cfg, device)

    # 2. Embedding precomputation
    cache_enabled = cfg.get('embedding_cache', {}).get('enabled', True)
    ckpt_path = model_info['checkpoint_path']
    cache_meta = {
        'model_config': model_config_name,
        'model_class': f"{model_cfg.get('module', '')}.{model_cfg.get('class', '')}",
        'checkpoint': os.path.abspath(ckpt_path),
        'checkpoint_mtime': round(os.path.getmtime(ckpt_path), 3) if os.path.exists(ckpt_path) else None,
        'num_nodes': model_info['num_nodes'],
        # Width of X: the degree channel (dataset.use_attr_degree) changes d -> d + 1, so cached
        # embeddings of a differently-built X must not be reused even with the same checkpoint.
        'num_features': model_info.get('num_features'),
        'time_steps': [int(t) for t in model_info['time_steps']],
        # The graph is sparser when min_degree > 0, so the cached embeddings are no longer valid
        # for another min_degree even with the same checkpoint and node count.
        'min_degree': model_info.get('min_degree', 0),
    }
    embedding_cache = EmbeddingCache(
        model, model_info['year_to_idx'], len(model_info['time_steps']), meta=cache_meta
    )

    if cache_enabled and embedding_cache.is_compatible():
        embedding_cfg = cfg.get('embedding_cache', {})
        force = embedding_cfg.get('force_recompute', False)
        save_path = embedding_cfg.get('save_path', '') or os.path.join(
            ROOT_DIR, 'results', cfg.get('output_subdir', 'default'),
            'imputation_embeddings.pt'
        )
        embeddings = embedding_cache.precompute_embeddings(
            force=force, save_path=save_path, device=device
        )
    elif cache_enabled and not embedding_cache.is_compatible():
        print("[INFO] Current model does not support embedding precomputation, will use per-batch forward inference")

    # 3. Create predictor
    predictor = ImputationPredictorFast(
        model=model,
        full_dataset=full_dataset,
        cfg=cfg,
        model_info=model_info,
        device=device,
        embedding_cache=embedding_cache if cache_enabled else None,
    )

    if cache_enabled and embedding_cache._cached is not None:
        predictor._dynamic_embeddings = embedding_cache._cached.to(device)
        print(f"[EmbeddingCache] Embeddings ready, will use precomputed embeddings for efficient prediction")

    # 4. Load industry network
    indus_network = predictor.load_industry_network()

    # 5. Run prediction and imputation
    predictor.run(indus_network)


if __name__ == "__main__":
    main()
