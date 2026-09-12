"""
Per-model threshold search script (based on Bootstrap results)

Difference from find_threshold.py:
    - find_threshold.py: ensemble average -> 1 set of thresholds (ensemble)
    - This script: each Bootstrap checkpoint computed independently -> B sets of thresholds + summary stats (mean +/- std)

Usage:
    python Prediction/find_threshold_per_model.py -b results/bootstrap/<run>/
    python Prediction/find_threshold_per_model.py -b results/bootstrap/<run>/ -d cuda:0 --save-results

Output:
    - threshold_analysis/per_model_results.json: per-model thresholds + summary stats
    - threshold_analysis/per_model_thresholds.png: threshold distribution plot (boxplot + scatter)
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

from utils import import_attr, resolve_auto_kwargs, resolve_device
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
    """Load the test set from the bootstrap output directory (positives + deterministically rebuilt negatives)"""
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

    # Load fixed test-set positives
    test_pos_path = os.path.join(bootstrap_dir, 'test_pos.csv')
    test_pos_df = pd.read_csv(test_pos_path)
    test_pos = list(test_pos_df.itertuples(index=False, name=None))
    print(f"  Test pos samples: {len(test_pos)}")

    # Create full_dataset
    print("\n[1/5] Creating full_dataset...")
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

    # Deterministically rebuild negatives
    print("\n[2/5] Rebuilding test negative samples...")
    rng = np.random.RandomState(seed_base)
    all_pos_raw = full_dataset.original_positive_samples
    all_pos = list(set(all_pos_raw))
    all_pos.sort(key=lambda s: (s[2], s[0], s[1]))

    neg_pool = _build_negative_pool(full_dataset, all_pos, rng)
    test_neg = _sample_fixed_negatives(
        full_dataset, test_pos, len(test_pos), neg_pool, rng
    )
    print(f"  Test pos: {len(test_pos)}, fixed neg: {len(test_neg)}")

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

    # Create test_dataset
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
    test_dataset.node_mapping = full_dataset.node_mapping
    test_dataset.reverse_node_mapping = full_dataset.reverse_node_mapping
    test_dataset.factset_edges = full_dataset.factset_edges

    # Load static graph
    print("\n[4/5] Loading static graph...")
    pyg_path = os.path.join(bootstrap_dir, 'dynamic_data.pyg')
    dynamic_data = torch.load(pyg_path, map_location='cpu', weights_only=False)
    print(f"  Nodes: {dynamic_data.num_nodes}, Edges: {dynamic_data.edge_index.shape[1]}")

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
    paths = sorted(glob.glob(pattern),
                   key=lambda p: int(p.split('_')[-1].replace('.pth', '')))
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-model inference, returns (prediction probabilities, labels)"""
    loader = PyGDataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    model.eval()
    all_probs = []
    all_labels = []

    for link_indices, current_times, labels in loader:
        link_indices = link_indices.to(device)
        current_times = current_times.to(device)

        predictions = model(link_indices, current_times)
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            predictions = torch.sigmoid(predictions)

        all_probs.append(predictions.cpu().numpy().flatten())
        all_labels.append(labels.numpy().flatten())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# Threshold search

def compute_thresholds(all_probs: np.ndarray, all_labels: np.ndarray) -> Dict[str, float]:
    """Compute the optimal threshold for a single model's predictions (no report printed)"""
    y_true = all_labels.astype(int)
    y_scores = all_probs.astype(float)

    valid = ~np.isnan(y_scores)
    y_true = y_true[valid]
    y_scores = y_scores[valid]

    # AUC
    auc_score = float(roc_auc_score(y_true, y_scores))

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    # PR curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = float(auc(recall_curve, precision_curve))

    # --- Youden's J ---
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    threshold_youden = float(roc_thresholds[best_j_idx])
    youden_j_value = float(j_scores[best_j_idx])
    fpr_youden = float(fpr[best_j_idx])
    tpr_youden = float(tpr[best_j_idx])

    # --- F1 maximization ---
    best_f1 = 0.0
    best_t_f1 = 0.5
    for t in np.arange(0.1, 0.9, 0.001):
        pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t_f1 = float(t)

    # --- F2 maximization ---
    best_f2 = 0.0
    best_t_f2 = 0.5
    for t in np.arange(0.1, 0.9, 0.001):
        pred = (y_scores >= t).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = float(f2)
            best_t_f2 = float(t)

    # --- P-R balance ---
    pr_gap = np.abs(precision_curve - recall_curve)
    best_pr_idx = np.argmin(pr_gap)
    threshold_pr_balance = float(pr_thresholds[best_pr_idx]) \
        if best_pr_idx < len(pr_thresholds) else 0.5
    pr_balance_gap = float(pr_gap[best_pr_idx])

    # --- Metrics at the Youden threshold ---
    pred_youden = (y_scores >= threshold_youden).astype(int)
    f1_youden = float(f1_score(y_true, pred_youden, zero_division=0))
    f2_youden = float(fbeta_score(y_true, pred_youden, beta=2, zero_division=0))

    # --- threshold=0.5 reference ---
    pred_05 = (y_scores >= 0.5).astype(int)
    p05 = float(precision_score(y_true, pred_05, zero_division=0))
    r05 = float(recall_score(y_true, pred_05, zero_division=0))
    f05 = float(f1_score(y_true, pred_05, zero_division=0))
    f2_05 = float(fbeta_score(y_true, pred_05, beta=2, zero_division=0))

    return {
        'auc': auc_score,
        'pr_auc': pr_auc,
        'youden_j': threshold_youden,
        'youden_j_stat': youden_j_value,
        'f1_max': best_t_f1,
        'f1_at_f1max': best_f1,
        'f2_max': best_t_f2,
        'f2_at_f2max': best_f2,
        'prec_rec_balance': threshold_pr_balance,
        'pr_balance_gap': pr_balance_gap,
        'fpr_at_youden': fpr_youden,
        'tpr_at_youden': tpr_youden,
        'f1_at_youden': f1_youden,
        'f2_at_youden': f2_youden,
        'precision_at_05': p05,
        'recall_at_05': r05,
        'f1_at_05': f05,
        'f2_at_05': f2_05,
    }


# Summary statistics

def summarize_per_model_results(
    per_model: List[Dict[str, float]],
    model_indices: List[int],
) -> Dict:
    """Compute summary stats such as mean +/- std over the B sets of thresholds"""
    # Only compute stats over numeric keys (skip metadata fields such as model_index, checkpoint)
    numeric_keys = [k for k, v in per_model[0].items() if isinstance(v, (int, float, np.floating, np.integer))]
    summary = {}

    for key in numeric_keys:
        values = np.array([m[key] for m in per_model], dtype=np.float64)
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)),  # sample standard deviation
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.max(values) - np.min(values)),
            'values': [float(v) for v in values],
        }

    return summary


def print_summary(summary: Dict):
    """Print summary statistics"""
    key_metrics = [
        'auc', 'pr_auc',
        'youden_j', 'youden_j_stat',
        'f1_max', 'f2_max', 'prec_rec_balance',
        'f1_at_youden', 'f2_at_youden',
        'f1_at_05',
    ]

    print("\nPer-Model Threshold Summary (across all Bootstrap iterations)")
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print("-" * 80)

    for key in key_metrics:
        if key not in summary:
            continue
        s = summary[key]
        print(f"{key:<25} {s['mean']:>10.4f} {s['std']:>10.4f} "
              f"{s['median']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['range']:>10.4f}")

    print(f"\n  Recommended threshold: {summary['youden_j']['mean']:.4f} "
          f"± {summary['youden_j']['std']:.4f}  (mean ± std)")
    print(f"  Ensemble: youden_j threshold varies across bootstrap models by "
          f"{summary['youden_j']['range']:.4f}")
    print()


# Visualization: threshold distribution

def plot_per_model_thresholds(
    summary: Dict,
    output_dir: str,
    model_name: str = "",
):
    """Plot the distribution of each threshold metric across the B models"""
    os.makedirs(output_dir, exist_ok=True)

    # Select the threshold-type metrics to display
    threshold_keys = [
        'youden_j', 'f1_max', 'f2_max', 'prec_rec_balance'
    ]
    threshold_labels = [
        "Youden's J", 'F1 Max', 'F2 Max', 'P-R Balance'
    ]

    # Extract data
    data = []
    labels = []
    for key, label in zip(threshold_keys, threshold_labels):
        if key in summary:
            data.append(summary[key]['values'])
            labels.append(label)

    if not data:
        print("No threshold data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              gridspec_kw={'width_ratios': [1.5, 1, 1]})
    model_label = f"\n({model_name})" if model_name else ""

    ax = axes[0]
    positions = range(1, len(data) + 1)

    bp = ax.boxplot(data, positions=positions, widths=0.4,
                     patch_artist=True, showfliers=False,
                     medianprops={'color': 'black', 'linewidth': 1.5})

    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    # Scatter (one point per model)
    for i, (d, pos) in enumerate(zip(data, positions)):
        jitter = np.random.uniform(-0.1, 0.1, size=len(d))
        ax.scatter(np.full_like(d, pos) + jitter, d,
                   alpha=0.6, s=30, color=colors[i], edgecolors='white',
                   linewidth=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Threshold Value', fontsize=11)
    ax.set_title(f'Threshold Distribution Across B Models{model_label}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    if 'auc' in summary:
        auc_vals = summary['auc']['values']
        ax.boxplot([auc_vals], widths=0.3, patch_artist=True,
                    boxprops={'facecolor': '#3498db', 'alpha': 0.3},
                    showfliers=False)
        jitter = np.random.uniform(-0.08, 0.08, size=len(auc_vals))
        ax.scatter(np.ones_like(auc_vals) + jitter, auc_vals,
                   alpha=0.6, s=30, color='#3498db', edgecolors='white',
                   linewidth=0.5)
        ax.axhline(y=np.mean(auc_vals), color='red', linestyle='--',
                   alpha=0.6, linewidth=1, label=f"Mean={np.mean(auc_vals):.4f}")
        ax.set_xticklabels(['ROC AUC'], fontsize=10)
        ax.set_ylabel('AUC', fontsize=11)
        ax.set_title(f'AUC Variation{model_label}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    if 'youden_j' in summary and 'youden_j_stat' in summary:
        t_vals = summary['youden_j']['values']
        j_vals = summary['youden_j_stat']['values']
        ax.scatter(t_vals, j_vals, alpha=0.7, s=40, color='#e74c3c',
                   edgecolors='white', linewidth=0.5)
        for i, (tx, jx) in enumerate(zip(t_vals, j_vals)):
            ax.annotate(str(i), (tx, jx), fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel("Youden's J Threshold", fontsize=11)
        ax.set_ylabel("Youden's J Statistic", fontsize=11)
        ax.set_title(f'Threshold vs J-Statistic{model_label}',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=2)
    fig_path = os.path.join(output_dir, 'per_model_thresholds.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Per-model threshold chart saved: {fig_path}")

    return fig_path


# Main entry

def main():
    parser = argparse.ArgumentParser(
        description='Per-model threshold search (each Bootstrap checkpoint independently)'
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
        help='Save per-model results and raw predictions'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Inference batch size (default: from bootstrap config)'
    )
    parser.add_argument(
        '--max-models', type=int, default=None,
        help='Limit to first N checkpoints (for quick testing)'
    )

    args = parser.parse_args()

    bootstrap_dir = os.path.abspath(args.bootstrap_dir)
    if not os.path.isdir(bootstrap_dir):
        raise NotADirectoryError(f"bootstrap directory not found: {bootstrap_dir}")

    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    print("=" * 60)
    print("Per-Model Threshold Search (Bootstrap)")
    print("=" * 60)
    print(f"Bootstrap dir: {bootstrap_dir}")
    print(f"Device:        {device}")
    print(f"Strategy:      Each checkpoint → independent threshold")
    print()

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

    # 3. Discover checkpoints
    checkpoint_paths = discover_checkpoints(bootstrap_dir)
    if args.max_models:
        checkpoint_paths = checkpoint_paths[:args.max_models]
    B = len(checkpoint_paths)
    print(f"\nFound {B} checkpoint(s):")
    for p in checkpoint_paths:
        print(f"  - {os.path.basename(p)}")

    # 4. Per-model inference + threshold computation
    per_model_results = []
    per_model_probs = {}  # model_idx -> probs (optional save)
    all_labels = None

    total_skipped = 0
    print(f"\n[5/5] Per-model inference & threshold search ({B} models)...")

    for b_idx, ckpt_path in enumerate(checkpoint_paths):
        ckpt_name = os.path.basename(ckpt_path)
        model_idx = int(ckpt_name.split('_')[-1].replace('.pth', ''))

        print(f"\n  [{b_idx+1}/{B}] {ckpt_name}")
        t_start = time.time()

        # Load weights
        skipped = load_checkpoint(model, ckpt_path, device)
        total_skipped += skipped
        if skipped > 0:
            print(f"    Skipped {skipped} mismatched param keys")

        # Inference
        probs, labels = run_single_model_inference(model, test_dataset, batch_size, device)
        if all_labels is None:
            all_labels = labels
        elapsed = time.time() - t_start
        print(f"    Inference: {elapsed:.1f}s, samples: {len(probs):,}")

        # Compute thresholds
        thresholds = compute_thresholds(probs, labels)
        thresholds['model_index'] = model_idx
        thresholds['checkpoint'] = ckpt_name
        per_model_results.append(thresholds)

        if args.save_results:
            per_model_probs[model_idx] = probs

        # Brief output
        print(f"    AUC={thresholds['auc']:.4f}, "
              f"Youden t={thresholds['youden_j']:.4f} (J={thresholds['youden_j_stat']:.4f}), "
              f"F1 t={thresholds['f1_max']:.4f}, F2 t={thresholds['f2_max']:.4f}")

    if total_skipped > 0:
        print(f"\n  Total skipped {total_skipped} param keys across {B} models")

    # 5. Summary statistics
    model_indices = [m['model_index'] for m in per_model_results]
    summary = summarize_per_model_results(per_model_results, model_indices)

    # If an ensemble threshold_results.json exists, read it for comparison
    ensemble_path = os.path.join(bootstrap_dir, 'threshold_analysis', 'threshold_results.json')
    if os.path.exists(ensemble_path):
        with open(ensemble_path, 'r', encoding='utf-8') as f:
            ensemble_results = json.load(f)
        summary['ensemble_results'] = {k: v for k, v in ensemble_results.items()
                                       if isinstance(v, (int, float))}
        print("  (Found ensemble threshold_results.json for comparison)")

    print_summary(summary)

    # 6. Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(bootstrap_dir, 'threshold_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 7. Visualization
    bootstrap_name = os.path.basename(bootstrap_dir.rstrip('/\\'))
    plot_per_model_thresholds(
        summary, output_dir, model_name=bootstrap_name
    )

    # 8. Save results
    if args.save_results:
        # Clean numpy types
        def clean_value(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, list):
                return [clean_value(x) for x in v]
            return v

        output_data = {
            'bootstrap_dir': bootstrap_dir,
            'model_config': model_config_name,
            'num_checkpoints': B,
            'strategy': 'per_model_independent',
            'per_model': [{k: clean_value(v) for k, v in m.items()}
                          for m in per_model_results],
            'summary': {k: {sk: clean_value(sv) for sk, sv in v.items()}
                        for k, v in summary.items()},
        }

        json_path = os.path.join(output_dir, 'per_model_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Per-model results saved: {json_path}")

        # Save the raw predictions of each model
        if per_model_probs:
            npz_path = os.path.join(output_dir, 'per_model_predictions.npz')
            save_dict = {f'model_{idx}': probs
                         for idx, probs in per_model_probs.items()}
            save_dict['labels'] = all_labels
            np.savez_compressed(npz_path, **save_dict)
            print(f"Per-model predictions saved: {npz_path}")

    # 9. Recommended thresholds
    print(f"\nRecommended threshold (mean of {B} per-model thresholds):")
    print(f"  Youden's J:  {summary['youden_j']['mean']:.4f} ± {summary['youden_j']['std']:.4f}")
    print(f"  F1 Max:      {summary['f1_max']['mean']:.4f} ± {summary['f1_max']['std']:.4f}")
    print(f"  F2 Max:      {summary['f2_max']['mean']:.4f} ± {summary['f2_max']['std']:.4f}")
    print()

    # Compare with the ensemble
    if 'ensemble_results' in summary:
        ens = summary['ensemble_results']
        print("Comparison with ensemble results:")
        print(f"  Ensemble youden_j threshold: {ens.get('youden_j', 'N/A'):.4f}")
        print(f"  Per-model mean:              {summary['youden_j']['mean']:.4f}")
        print(f"  Difference:                  "
              f"{summary['youden_j']['mean'] - ens.get('youden_j', 0):.4f}")
        print()


if __name__ == "__main__":
    main()
