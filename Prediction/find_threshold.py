"""
Optimal threshold search script (based on Bootstrap results)

Loads a Bootstrap output directory (fixed test set + B model checkpoints) and
computes the optimal threshold via ensemble averaging, then outputs analysis plots.

Usage:
    python Prediction/find_threshold.py --bootstrap-dir results/bootstrap/<run>/
    python Prediction/find_threshold.py -b results/bootstrap/<run>/ -d cuda:0 --save-results

Dependencies:
    pip install matplotlib scikit-learn
"""

import sys
import os
import argparse
import json
import time
import logging
import importlib
import inspect
import glob
from typing import Any, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils import deep_merge, import_attr, resolve_auto_kwargs, resolve_device
from Training.config_loader import load_config

# Suppress Neo4j logs
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, fbeta_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Data.company_dataset import (
    CompanySupplyDataset,
    dataset_feature_kwargs,
    _build_negative_pool,
    _sample_fixed_negatives,
    _load_csv_negatives,
    _split_8_1_1,
)


# Bootstrap data loading

def load_bootstrap_test_set(bootstrap_dir: str) -> dict:
    """Load the test set from the bootstrap output directory (positives + deterministically rebuilt negatives)

    Returns:
        dict with keys:
            test_pos:       list of (src, tgt, year) tuples
            test_neg:       list of (src, tgt, year) tuples (deterministically rebuilt)
            test_dataset:   CompanySupplyDataset (with positives and negatives)
            full_dataset:   CompanySupplyDataset (metadata)
            dynamic_data:   PyG Data (static graph)
            node_mapping:   dict
            reverse_node_mapping: dict
            seed_base:      int
            num_bootstrap:  int
    """
    # 1. Load bootstrap config
    config_path = os.path.join(bootstrap_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"bootstrap config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        bootstrap_cfg = yaml.safe_load(f)

    model_config_name = bootstrap_cfg['model']['config']
    data_cfg = bootstrap_cfg['data']
    seed_base = bootstrap_cfg['bootstrap']['seed_base']
    num_bootstrap = bootstrap_cfg['bootstrap']['num_iterations']

    print(f"\nLoading Bootstrap data: {bootstrap_dir}")
    print(f"  Model config: {model_config_name}")
    print(f"  seed_base:    {seed_base}")
    print(f"  B iterations: {num_bootstrap}")
    print(f"  Data params:  test={data_cfg['test_ratio']}, val={data_cfg['val_ratio']}, "
          f"neg_ratio={data_cfg['negative_ratio']}, min_degree={data_cfg['min_degree']}")

    # 2. Load fixed test-set positives
    test_pos_path = os.path.join(bootstrap_dir, 'test_pos.csv')
    if not os.path.exists(test_pos_path):
        raise FileNotFoundError(f"test_pos.csv not found: {test_pos_path}")
    test_pos_df = pd.read_csv(test_pos_path)
    test_pos = list(test_pos_df.itertuples(index=False, name=None))
    print(f"  Test pos samples: {len(test_pos)}")

    # 3. Create full_dataset (only for fetching metadata)
    print("\n[1/5] Creating full_dataset (metadata: company IDs, industries, known edges)...")
    full_dataset = CompanySupplyDataset(
        negative_ratio=0,
        embedding_name=data_cfg['embedding_name'],
        neg_dir=None,
        toy_mode=False,
        min_degree=data_cfg.get('min_degree', 2),
        source_filter=data_cfg.get('source_filter', 'semi'),
        other_possible_fill=data_cfg.get('other_possible_fill', 0.0),
        filter_factset_neg=data_cfg.get('filter_factset_neg', False),
        intra_industry_neg=data_cfg.get('intra_industry_neg', True),
        use_pred_neg=False,
        **dataset_feature_kwargs(data_cfg),
    )
    print(f"  Companies: {len(full_dataset.company_ids)}")
    print(f"  FactSet edges: {len(full_dataset.factset_edges)}")
    print(f"  available_years: {full_dataset.available_years}")

    # 4. Deterministically rebuild negatives (exactly as during bootstrap training)
    print("\n[2/5] Rebuilding test negative samples (deterministic, same as bootstrap training)...")
    rng = np.random.RandomState(seed_base)
    # Deduplicate and sort all positives (consistent with company_dataset.create_bootstrap_datasets)
    all_pos_raw = full_dataset.original_positive_samples
    all_pos = list(set(all_pos_raw))
    all_pos.sort(key=lambda s: (s[2], s[0], s[1]))

    neg_pool = _build_negative_pool(full_dataset, all_pos, rng)
    print(f"  Negative pool size: {len(neg_pool)}")

    test_neg = _sample_fixed_negatives(
        full_dataset, test_pos, len(test_pos), neg_pool, rng
    )
    print(f"  Test fixed neg samples: {len(test_neg)}")

    # CSV predefined negatives: the 0.1 test share, rebuilt with exactly the same procedure as in
    # company_dataset.create_bootstrap_datasets (same helpers, same seed, same ratios). Older run
    # configs predate this key, so the default is False - i.e. no CSV was used back then.
    use_pred_neg = data_cfg.get('use_pred_neg', False)
    csv_neg = _load_csv_negatives(
        full_dataset, all_pos, use_pred_neg,
        data_cfg.get('neg_dir', None), data_cfg.get('filter_factset_neg', False)
    )
    if csv_neg:
        _, _, csv_neg_test = _split_8_1_1(
            csv_neg, data_cfg['test_ratio'], data_cfg['val_ratio'], seed_base,
            label='CSV negatives'
        )
        csv_neg_test = [t for t in csv_neg_test if t not in set(test_neg)]
        print(f"  Test CSV predefined negatives: {len(csv_neg_test)}")
    else:
        csv_neg_test = []

    # 5. Create test_dataset
    # Static-only evaluation set (1:1 + the CSV share), exactly as in
    # company_dataset.create_bootstrap_datasets: `negative_ratio=0` disables the dynamic negatives,
    # so len(dataset) == 2 * len(test_pos) + len(csv_neg_test).
    print("\n[3/5] Creating test CompanySupplyDataset...")
    test_dataset = CompanySupplyDataset(
        negative_ratio=0,
        embedding_name=data_cfg['embedding_name'],
        neg_dir=None,
        toy_mode=False,
        positive_samples=list(test_pos),
        fixed_neg_data=test_neg,
        predifined_neg_data=csv_neg_test,
        min_degree=data_cfg.get('min_degree', 2),
        source_filter=data_cfg.get('source_filter', 'semi'),
        other_possible_fill=data_cfg.get('other_possible_fill', 0.0),
        filter_factset_neg=data_cfg.get('filter_factset_neg', False),
        intra_industry_neg=False,
        use_pred_neg=use_pred_neg,
    )
    # Share metadata
    test_dataset.node_mapping = full_dataset.node_mapping
    test_dataset.reverse_node_mapping = full_dataset.reverse_node_mapping
    test_dataset.factset_edges = full_dataset.factset_edges

    print(f"  Test total samples: {len(test_dataset)} "
          f"(pos={len(test_pos)}, neg={len(test_dataset) - len(test_pos)}, "
          f"fixed neg={len(test_neg)}, csv neg={len(csv_neg_test) if use_pred_neg else 0}, dynamic=0)")

    # 6. Load static graph
    print("\n[4/5] Loading static graph...")
    pyg_path = os.path.join(bootstrap_dir, 'dynamic_data.pyg')
    if not os.path.exists(pyg_path):
        raise FileNotFoundError(f"dynamic_data.pyg not found: {pyg_path}")
    dynamic_data = torch.load(pyg_path, map_location='cpu', weights_only=False)
    print(f"  Nodes: {dynamic_data.num_nodes}")
    print(f"  Edges: {dynamic_data.edge_index.shape[1]}")

    return {
        'test_pos': test_pos,
        'test_neg': test_neg,
        'test_dataset': test_dataset,
        'full_dataset': full_dataset,
        'dynamic_data': dynamic_data,
        'node_mapping': full_dataset.node_mapping,
        'reverse_node_mapping': full_dataset.reverse_node_mapping,
        'seed_base': seed_base,
        'num_bootstrap': num_bootstrap,
        'model_config_name': model_config_name,
        'data_cfg': data_cfg,
        'bootstrap_cfg': bootstrap_cfg,
    }


# Model loading

def load_model_architecture(model_config_name: str, dynamic_data, device: torch.device) -> nn.Module:
    """Load model architecture by config name (random initialization)"""
    model_config_dir = os.path.join(ROOT_DIR, 'Models', 'configs')
    yaml_path = os.path.join(model_config_dir, f"{model_config_name}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Model config not found: {yaml_path}")

    # Shared model parts (e.g. the node-feature extractor) live in Training/common_config.yaml, so
    # merge it exactly like Training/config_loader.load_config does. Reading the model YAML alone
    # would silently fall back to the code defaults for those keys.
    model_cfg = load_config(model_config_name)

    model_params = model_cfg['model']
    ModelClass = import_attr(model_params['module'], model_params['class'])

    time_steps = sorted(dynamic_data.edge_time.unique().tolist())
    # Ensure the full year range
    for y in range(2013, 2026):
        if y not in time_steps:
            time_steps.append(y)
    time_steps = sorted(time_steps)

    auto_context = {
        'num_features': dynamic_data.x.size(1),
        'num_nodes': dynamic_data.num_nodes,
        'time_steps': time_steps,
        'hidden_dims': dynamic_data.x.size(1),
        'device': device,
    }
    model_kwargs = resolve_auto_kwargs(model_params.get('kwargs', {}), auto_context)

    init_params = inspect.signature(ModelClass.__init__).parameters
    if 'dynamic_data' in init_params:
        model = ModelClass(dynamic_data=dynamic_data, **model_kwargs)
    else:
        model = ModelClass(**model_kwargs)

    return model.to(device)


def discover_checkpoints(bootstrap_dir: str) -> List[str]:
    """Discover all best_model_N.pth under the bootstrap output dir"""
    pattern = os.path.join(bootstrap_dir, 'best_model_*.pth')
    paths = sorted(glob.glob(pattern), key=lambda p: int(p.split('_')[-1].replace('.pth', '')))
    if not paths:
        raise FileNotFoundError(f"No best_model_*.pth found in: {bootstrap_dir}")
    return paths


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """Load a single checkpoint's weights into the model, skipping mismatched keys"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped += 1

    model.load_state_dict(filtered, strict=False)
    return skipped


# Inference

@torch.no_grad()
def run_single_model_inference(
    model: nn.Module,
    test_dataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Single-model inference, returns the prediction probability array [N]"""
    loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs = []

    for link_indices, current_times, labels in tqdm(loader, desc="  Inference", leave=False):
        link_indices = link_indices.to(device)
        current_times = current_times.to(device)

        predictions = model(link_indices, current_times)
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            predictions = torch.sigmoid(predictions)

        all_probs.append(predictions.cpu().numpy().flatten())

    return np.concatenate(all_probs)


def run_ensemble_inference(
    model: nn.Module,
    checkpoint_paths: List[str],
    test_dataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensemble inference: run inference on every checkpoint separately, average the probabilities

    Returns:
        (all_probs_mean, all_labels):
            all_probs_mean: [N] mean predicted probability (ensemble)
            all_labels:     [N] ground-truth labels
    """
    B = len(checkpoint_paths)
    print(f"\n[5/5] Ensemble inference (averaging {B} models)...")

    # First extract all labels (identical for every model)
    loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_labels = []
    for _, _, labels in loader:
        all_labels.append(labels.numpy().flatten())
    all_labels = np.concatenate(all_labels)

    # Run inference on each checkpoint separately
    all_model_probs = np.zeros((B, len(all_labels)), dtype=np.float32)
    total_skipped = 0

    for b_idx, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n  [{b_idx+1}/{B}] {os.path.basename(ckpt_path)}")
        skipped = load_checkpoint(model, ckpt_path, device)
        total_skipped += skipped
        if skipped > 0:
            print(f"    Skipped {skipped} mismatched param keys")

        probs = run_single_model_inference(model, test_dataset, batch_size, device)
        all_model_probs[b_idx] = probs.astype(np.float32)

    if total_skipped > 0:
        print(f"\n  Total skipped {total_skipped} param keys across {B} models")

    # Ensemble average
    all_probs_mean = all_model_probs.mean(axis=0)

    print(f"\nEnsemble inference complete:")
    print(f"  Total samples:     {len(all_labels):,}")
    print(f"  Positive:          {int(all_labels.sum()):,}")
    print(f"  Negative:          {int(len(all_labels) - all_labels.sum()):,}")
    print(f"  Ensemble size:     {B}")
    print(f"  Mean prediction:   {all_probs_mean.mean():.4f} +/- {all_probs_mean.std():.4f}")

    # Also report prediction agreement across models
    model_corrs = np.corrcoef(all_model_probs)
    mean_corr = (model_corrs.sum() - B) / (B * (B - 1))  # exclude the diagonal
    print(f"  Mean inter-model correlation: {mean_corr:.4f} (higher = more consistent)")

    return all_probs_mean, all_labels


# Threshold search

def find_optimal_thresholds(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
) -> Dict[str, float]:
    """Search for the optimal threshold using multiple methods"""
    y_true = all_labels.astype(int)
    y_scores = all_probs.astype(float)

    valid = ~np.isnan(y_scores)
    y_true = y_true[valid]
    y_scores = y_scores[valid]

    print("\nThreshold Search and Evaluation")

    auc_score = roc_auc_score(y_true, y_scores)
    print(f"\nROC AUC: {auc_score:.4f}")

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    print(f"PR AUC:  {pr_auc:.4f}")

    # --- Method 1: Youden's J ---
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    threshold_youden = float(roc_thresholds[best_j_idx])
    youden_j_value = float(j_scores[best_j_idx])  # the actual Youden's J statistic
    fpr_youden = float(fpr[best_j_idx])
    tpr_youden = float(tpr[best_j_idx])

    # --- Method 2: F1 maximization ---
    best_f1 = 0.0
    best_t_f1 = 0.5
    for t in np.arange(0.1, 0.9, 0.001):
        pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t_f1 = float(t)

    # --- Method 3: F2 maximization (recall-oriented) ---
    best_f2 = 0.0
    best_t_f2 = 0.5
    for t in np.arange(0.1, 0.9, 0.001):
        pred = (y_scores >= t).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = float(f2)
            best_t_f2 = float(t)

    # --- Method 4: Precision-Recall balance ---
    pr_gap = np.abs(precision_curve - recall_curve)
    best_pr_idx = np.argmin(pr_gap)
    threshold_pr_balance = float(pr_thresholds[best_pr_idx]) \
        if best_pr_idx < len(pr_thresholds) else 0.5
    pr_balance_gap = float(pr_gap[best_pr_idx])  # P-R gap (smaller is better)

    # --- Metrics at the Youden threshold ---
    pred_youden = (y_scores >= threshold_youden).astype(int)
    f1_youden = float(f1_score(y_true, pred_youden, zero_division=0))
    f2_youden = float(fbeta_score(y_true, pred_youden, beta=2, zero_division=0))

    # --- Threshold=0.5 reference ---
    pred_05 = (y_scores >= 0.5).astype(int)
    p05 = float(precision_score(y_true, pred_05, zero_division=0))
    r05 = float(recall_score(y_true, pred_05, zero_division=0))
    f05 = float(f1_score(y_true, pred_05, zero_division=0))
    f2_05 = float(fbeta_score(y_true, pred_05, beta=2, zero_division=0))

    pred_0452 = (y_scores >= 0.452).astype(int)
    p_0452 = float(precision_score(y_true, pred_0452, zero_division=0))
    r_0452 = float(recall_score(y_true, pred_0452, zero_division=0))
    f1_0452 = float(f1_score(y_true, pred_0452, zero_division=0))

    results = {
        'auc': auc_score,
        'pr_auc': pr_auc,
        'youden_j': threshold_youden,                # threshold at Youden's J optimum
        'youden_j_stat': youden_j_value,             # Youden's J statistic (TPR - FPR)
        'f1_max': best_t_f1,
        'f2_max': best_t_f2,
        'prec_rec_balance': threshold_pr_balance,    # threshold at P-R balance
        'pr_balance_gap': pr_balance_gap,            # P-R gap
        'fpr_at_youden': fpr_youden,
        'tpr_at_youden': tpr_youden,
        'f1_at_youden': f1_youden,
        'f2_at_youden': f2_youden,
        'precision_at_05': p05,
        'recall_at_05': r05,
        'f1_at_05': f05,
        'f2_at_05': f2_05,
        'precision_at_0452': p_0452,
        'recall_at_0452': r_0452,
        'f1_at_0452': f1_0452,
        'f1_at_f1max': best_f1,
        'f2_at_f2max': best_f2,
    }

    # --- Print report ---
    print(f"\n{'Method':<25} {'Thresh':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 65)

    def _metrics_at(t_val):
        pred = (y_scores >= t_val).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        return p, r, f

    methods = [
        ("Youden's J", threshold_youden),
        ("F1 Max", best_t_f1),
        ("F2 Max", best_t_f2),
        ("P-R Balance", threshold_pr_balance),
        ("Fixed 0.500", 0.5),
        ("Current 0.452", 0.452),
    ]

    for name, t_val in methods:
        p, r, f = _metrics_at(t_val)
        print(f"{name:<25} {t_val:>8.4f} {f:>8.4f} {p:>8.4f} {r:>8.4f}")

    print(f"\nRecommended imputation threshold: {threshold_youden:.4f} (Youden's J)")
    print(f"  FPR={fpr_youden:.4f}, TPR={tpr_youden:.4f} at this threshold")
    print(f"  Alternative: F1-optimal={best_t_f1:.4f} (F1={best_f1:.4f})")
    print(f"  Alternative: F2-optimal={best_t_f2:.4f} (F2={best_f2:.4f})")

    return results


# Visualization

def plot_threshold_analysis(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    threshold_results: Dict[str, float],
    output_dir: str,
    model_name: str = "",
):
    """Generate threshold analysis plots: ROC, F1 curve, score distribution, metrics summary"""
    y_true = all_labels.astype(int)
    y_scores = all_probs.astype(float)
    valid = ~np.isnan(y_scores)
    y_true = y_true[valid]
    y_scores = y_scores[valid]

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    model_label = f" ({model_name})" if model_name else ""

    ax = axes[0, 0]
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    auc_score = threshold_results['auc']

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random')

    t_youden = threshold_results['youden_j']
    fpr_y = threshold_results['fpr_at_youden']
    tpr_y = threshold_results['tpr_at_youden']
    ax.plot(fpr_y, tpr_y, 'ro', markersize=8,
            label=f"Youden's J (t={t_youden:.4f})")

    t_f1 = threshold_results['f1_max']
    fpr_idx = np.argmin(np.abs(roc_thresholds - t_f1))
    fpr_f1 = fpr[fpr_idx]
    tpr_f1 = tpr[fpr_idx]
    ax.plot(fpr_f1, tpr_f1, 'g^', markersize=8,
            label=f"F1 Max (t={t_f1:.4f})")

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve{model_label}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    thresholds_grid = np.arange(0.05, 0.95, 0.005)
    f1_scores = []
    prec_scores = []
    rec_scores = []
    for t in thresholds_grid:
        pred = (y_scores >= t).astype(int)
        f1_scores.append(f1_score(y_true, pred, zero_division=0))
        prec_scores.append(precision_score(y_true, pred, zero_division=0))
        rec_scores.append(recall_score(y_true, pred, zero_division=0))

    ax.plot(thresholds_grid, f1_scores, 'b-', linewidth=2, label='F1')
    ax.plot(thresholds_grid, prec_scores, 'g--', linewidth=1.5, alpha=0.7, label='Precision')
    ax.plot(thresholds_grid, rec_scores, 'r--', linewidth=1.5, alpha=0.7, label='Recall')

    ax.axvline(x=t_f1, color='g', linestyle=':', alpha=0.7)
    ax.axvline(x=t_youden, color='r', linestyle=':', alpha=0.7)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metrics vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes[1, 0]
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    bins = np.linspace(0, 1, 51)
    ax.hist(pos_scores, bins=bins, alpha=0.6, color='blue',
            label=f'Positive (n={len(pos_scores):,})', density=True)
    ax.hist(neg_scores, bins=bins, alpha=0.6, color='red',
            label=f'Negative (n={len(neg_scores):,})', density=True)

    for t_val, color, label in [
        (t_youden, 'red', f"Youden\n({t_youden:.4f})"),
        (t_f1, 'green', f"F1\n({t_f1:.4f})"),
    ]:
        ax.axvline(x=t_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        ax.text(t_val + 0.01, ax.get_ylim()[1] * 0.85, label,
                color=color, fontsize=8, fontweight='bold')

    ax.set_xlabel('Prediction Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Score Distribution (pos vs neg)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    ax.axis('off')

    methods = [
        ("Youden's J", threshold_results['youden_j']),
        ("F1 Max", threshold_results['f1_max']),
        ("F2 Max", threshold_results['f2_max']),
        ("P-R Balance", threshold_results['prec_rec_balance']),
        ("Fixed 0.500", 0.5),
        ("Current 0.452", 0.452),
    ]

    table_data = []
    for name, t_val in methods:
        pred = (y_scores >= t_val).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        table_data.append([name, f"{t_val:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])

    cols = ['Method', 'Threshold', 'Precision', 'Recall', 'F1']
    table = ax.table(cellText=table_data, colLabels=cols,
                     cellLoc='center', loc='center',
                     colWidths=[0.18, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for j in range(len(cols)):
        table[(1, j)].set_facecolor('#fff2cc')

    ax.set_title(f'AUC={auc_score:.4f}  |  PR-AUC={threshold_results.get("pr_auc", 0):.4f}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout(pad=2)
    fig_path = os.path.join(output_dir, 'threshold_analysis.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nChart saved: {fig_path}")

    return fig_path


# Main entry

def main():
    parser = argparse.ArgumentParser(
        description='Optimal threshold search (Bootstrap ensemble)'
    )
    parser.add_argument(
        '--bootstrap-dir', '-b',
        required=True,
        help='Bootstrap v2 output dir (contains config.yaml, test_pos.csv, best_model_*.pth, dynamic_data.pyg)'
    )
    parser.add_argument(
        '--device', '-d', default='auto',
        help='Device (auto/cuda/cpu)'
    )
    parser.add_argument(
        '--output-dir', '-o', default='',
        help='Output dir (default: bootstrap-dir/threshold_analysis/)'
    )
    parser.add_argument(
        '--save-results', action='store_true',
        help='Save threshold results and raw predictions'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Inference batch size (default: from bootstrap config)'
    )

    args = parser.parse_args()

    bootstrap_dir = os.path.abspath(args.bootstrap_dir)
    if not os.path.isdir(bootstrap_dir):
        raise NotADirectoryError(f"bootstrap directory not found: {bootstrap_dir}")

    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    print("=" * 60)
    print("Optimal Threshold Search (Bootstrap Ensemble)")
    print("=" * 60)
    print(f"Bootstrap dir: {bootstrap_dir}")
    print(f"Device:        {device}")

    # 1. Load bootstrap test set
    bootstrap_data = load_bootstrap_test_set(bootstrap_dir)
    test_dataset = bootstrap_data['test_dataset']
    dynamic_data = bootstrap_data['dynamic_data'].to(device)
    model_config_name = bootstrap_data['model_config_name']

    batch_size = args.batch_size or \
        bootstrap_data['bootstrap_cfg']['training'].get('batch_size', 128)

    # 2. Create model architecture
    print(f"\nCreating model architecture: {model_config_name}")
    model = load_model_architecture(model_config_name, dynamic_data, device)

    # 3. Discover and load all bootstrap checkpoints
    checkpoint_paths = discover_checkpoints(bootstrap_dir)
    print(f"\nFound {len(checkpoint_paths)} checkpoint(s):")
    for p in checkpoint_paths:
        print(f"  - {os.path.basename(p)}")

    # 4. Ensemble inference
    all_probs, all_labels = run_ensemble_inference(
        model, checkpoint_paths, test_dataset, batch_size, device
    )

    # 5. Search for the optimal threshold
    threshold_results = find_optimal_thresholds(all_probs, all_labels)

    # 6. Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(bootstrap_dir, 'threshold_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 7. Visualization
    bootstrap_name = os.path.basename(bootstrap_dir.rstrip('/\\'))
    plot_threshold_analysis(
        all_probs, all_labels, threshold_results,
        output_dir, model_name=bootstrap_name
    )

    # 8. Save results (optional)
    if args.save_results:
        json_path = os.path.join(output_dir, 'threshold_results.json')
        save_data = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in threshold_results.items()}
        save_data['bootstrap_dir'] = bootstrap_dir
        save_data['model_config'] = model_config_name
        save_data['num_checkpoints'] = len(checkpoint_paths)
        save_data['num_samples'] = int(len(all_probs))
        save_data['num_positive'] = int(all_labels.sum())
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"Threshold results saved: {json_path}")

        scores_path = os.path.join(output_dir, 'predictions.npz')
        np.savez_compressed(scores_path,
                            probs=all_probs, labels=all_labels)
        print(f"Predictions saved: {scores_path}")

    # 9. Summary
    print(f"\nUsage: update threshold in imputation_common.yaml:")
    print(f"  prediction:")
    print(f"    threshold: {threshold_results['youden_j']:.4f}   # Youden's J optimal (recommended)")
    print(f"    # threshold: {threshold_results['f1_max']:.4f}   # F1 optimal (alternative)")
    print(f"    # threshold: {threshold_results['f2_max']:.4f}   # F2 optimal (alternative, recall-biased)")
    print()


if __name__ == "__main__":
    main()
