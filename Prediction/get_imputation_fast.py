"""
Edge-imputation prediction script based on industry upstream/downstream dependencies.

Batch-writes candidate edges whose prediction score exceeds the threshold back to
Neo4j (source='imputation'), using background writer threads + UNWIND batch MERGE
to reduce network round trips.

Which trained backbone is loaded is decided in Prediction/imputation_common.yaml:

    model.config_name   Models/configs/<name>.yaml (CLI --config overrides it)
    model.checkpoint    explicit .pth (highest priority)
    model.run_dir       training output dir, newest best_model.pth inside wins
    (both empty)        results/<output_subdir>/**/best_model.pth

Usage:
    python Prediction/get_imputation_fast.py                       # uses model.config_name in the YAML
    python Prediction/get_imputation_fast.py --config egcn          # CLI overrides the YAML
    python Prediction/get_imputation_fast.py --checkpoint path/to/model.pth

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
import yaml
from tqdm import tqdm

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
        time_positions = []
        for t_idx in time_indices:
            pos = self.year_to_idx.get(int(t_idx.item()), -1)
            time_positions.append(pos)
        time_positions = torch.tensor(time_positions, device=device, dtype=torch.long)
        valid_mask = time_positions >= 0
        if not valid_mask.any():
            return torch.tensor([], device=device)
        time_positions = time_positions[valid_mask]
        node_pairs_valid = node_pairs[valid_mask]
        u_indices = node_pairs_valid[:, 0].long()
        v_indices = node_pairs_valid[:, 1].long()
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

    _BATCH_WRITE_QUERY_TEMPLATE = """
        UNWIND $batch AS row
        MATCH (u:%(label)s), (v:%(label)s)
        WHERE id(u) = row.src AND id(v) = row.tgt
        MERGE (u)-[r:%(relation)s {
            year: row.year,
            probability: row.prob,
            source: '%(source)s',
            model: '%(model_name)s'
        }]->(v)
        RETURN count(r) as written
    """

    def __init__(
        self,
        label_name: str,
        relation_name: str,
        output_source: str,
        output_model: str,
        flush_size: int = 500,
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

        # Thread-safe queue
        self._queue: queue.Queue = queue.Queue(maxsize=100)
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
                probability: {item['prob']:.4f},
                source: '{self.output_source}',
                model: '{self.output_model}'
            }}]->(v)
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
        """Stop background threads and wait for the queue to drain"""
        self._running = False
        # Wait for the queue to drain
        if not self._queue.empty():
            print(f"\n[Writer] Waiting for write queue to drain ({self._queue.qsize()} pending)...")
        for t in self._workers:
            t.join(timeout=30)
        print(f"[Writer] Background writing complete: {self.total_written:,} rows total, "
              f"{self._flush_count} batch writes")


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
        self.flush_size = perf_cfg.get('flush_size', 500)
        self.query_threads = perf_cfg.get('query_threads', 8)

        # Embeddings
        self._dynamic_embeddings: Optional[torch.Tensor] = None
        self.reverse_mapping = full_dataset.reverse_node_mapping


    def precompute_embeddings(self, force: bool = False) -> Optional[torch.Tensor]:
        if self.cache is None or not self.cache.is_compatible():
            print("[INFO] Model does not support embedding precomputation, will use per-batch forward inference")
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
        """Fetch company lists for all industries concurrently with multiple threads"""
        print("\n[Preload] Fetching industry company info with multiple threads...")
        all_industries = set()
        for downstream_ind, upstream_list in indus_network.items():
            all_industries.add(downstream_ind)
            for item in upstream_list:
                if isinstance(item, list):
                    all_industries.update(item)
                else:
                    all_industries.add(item)

        result = {}
        with ThreadPoolExecutor(max_workers=self.query_threads) as executor:
            futures = {executor.submit(self._get_industry_companies, ind): ind
                       for ind in all_industries}
            for future in tqdm(as_completed(futures), total=len(futures), desc="query companies"):
                ind = futures[future]
                try:
                    companies = future.result()
                    if companies:
                        result[ind] = companies
                except Exception as e:
                    print(f"\n[WARN] Query for industry '{ind}' failed: {e}")

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
        """Batch-query already-existing semi edges"""
        from Data.neo4j_SPLC import Neo4jClient
        client = Neo4jClient()
        existing = set()
        result = client.execute_query(f"""
            MATCH (u:{self.label_name})-[r:{self.relation_name} {{source: 'semi'}}]->(v:{self.label_name})
            WHERE id(u) IN {upstream_neo4j_ids}
              AND id(v) IN {downstream_neo4j_ids}
              AND r.year IN {list(self.year_range)}
            RETURN id(u) as u_id, id(v) as v_id, r.year as year
        """)
        if result:
            for record in result:
                existing.add((record['u_id'], record['v_id'], record['year']))
        return existing


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

        reverse_mapping = self.full_dataset.reverse_node_mapping

        industry_companies = self._get_all_industry_companies(indus_network)
        if self.exclude_low_degree_nodes and self.min_degree > 0:
            industry_companies = self._filter_low_degree_companies(industry_companies)
        elif self.exclude_low_degree_nodes:
            print("[WARN] data.exclude_low_degree_nodes=true but data.min_degree <= 0: "
                  "nothing to exclude (set data.min_degree > 0)")

        total_estimate, _ = self.estimate_total_pairs(indus_network, industry_companies)

        writer = Neo4jBatchWriter(
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

                existing_set = set()
                if self.skip_existing:
                    up_neo4j = [reverse_mapping[uid] for uid in upstream_ids]
                    down_neo4j = [reverse_mapping[did] for did in downstream_ids]
                    existing_set = self._query_existing_edges(up_neo4j, down_neo4j)

                pending = []
                for src, tgt, year in self._generate_candidate_pairs(
                    upstream_ids, downstream_ids, self.year_range
                ):
                    src_neo4j = reverse_mapping.get(src)
                    tgt_neo4j = reverse_mapping.get(tgt)
                    if src_neo4j is None or tgt_neo4j is None:
                        continue
                    if self.skip_existing and (src_neo4j, tgt_neo4j, year) in existing_set:
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    pending.append({
                        'src': src, 'tgt': tgt,
                        'src_neo4j': src_neo4j, 'tgt_neo4j': tgt_neo4j,
                        'year': year,
                    })

                if not pending:
                    continue

                for i in range(0, len(pending), self.batch_size):
                    batch = pending[i:i + self.batch_size]

                    node_pairs = torch.tensor(
                        [[p['src'], p['tgt']] for p in batch],
                        device=self.device
                    )
                    time_indices = torch.tensor(
                        [p['year'] for p in batch],
                        device=self.device
                    )

                    probs = self.predict_batch(node_pairs, time_indices)
                    probs_cpu = probs.cpu().tolist()

                    for j, prob in enumerate(probs_cpu):
                        total_processed += 1
                        pbar.update(1)

                        if prob > self.threshold:
                            saved_count += 1
                            info = batch[j]
                            # Async write: push directly into the queue, do not wait for Neo4j
                            writer.enqueue(
                                info['src_neo4j'], info['tgt_neo4j'],
                                info['year'], prob
                            )

                    # Progress update
                    if total_processed > 0:
                        elapsed = time.time() - start_time
                        speed = total_processed / elapsed
                        remaining = total_estimate - total_processed - skipped_count
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

    args = parser.parse_args()

    print("=" * 60)
    print("Temporal GNN edge imputation prediction")
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
    print(f"Writers:      {perf_cfg_final.get('num_writers', 4)}")
    print(f"Batch write:  {perf_cfg_final.get('flush_size', 500)} rows/write")
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
