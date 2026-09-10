"""
Bootstrap v2 analysis script -- quantitative stability metrics based on a fixed test set.

Output metrics:
    5.1 Macro performance stability: mean +/- std and range of AUC/F1 across B models on the test set
    5.2 Micro individual-prediction stability: distribution and quantiles of the per-sample prediction-probability std across B models
    5.3 Prediction interval width: per-sample 95% prediction interval width (97.5% - 2.5% quantiles) and its mean

Usage:
    # All data parameters are read automatically from config.yaml in the output directory
    python Bootstraps/bootstrap_v2_analysis.py --dir results/bootstrap/<run>

    # CLI can override the device and batch size
    python Bootstraps/bootstrap_v2_analysis.py --dir results/bootstrap/xxx/ --device cuda:1 --batch_size 256
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import inspect
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from utils import import_attr, resolve_auto_kwargs, resolve_device


# Data extraction and inference

def extract_test_data_with_graph_indices(bootstrap_dir, batch_size=128):
    """
    Extract test-set data from the Bootstrap output directory, using graph indices (not neo4j ids).

    All data parameters are read automatically from config.yaml in the output directory, no manual configuration needed.
    """
    # Load the run config saved by the training script
    with open(os.path.join(bootstrap_dir, 'config.yaml'), 'r') as f:
        run_cfg = yaml.safe_load(f)

    from Data.company_dataset import create_bootstrap_datasets
    from torch_geometric.loader import DataLoader as PyGDataLoader

    # Read data parameters (must match those used at training time)
    data_cfg = run_cfg['data']
    seed_base = run_cfg['bootstrap']['seed_base']
    toy_mode = run_cfg['debug'].get('toy_mode', False)

    bootstrap_data = create_bootstrap_datasets(
        negative_ratio=data_cfg['negative_ratio'],
        embedding_name=data_cfg.get('embedding_name', 'embedding'),
        test_ratio=data_cfg['test_ratio'],
        val_ratio=data_cfg['val_ratio'],
        random_state=seed_base,
        min_degree=data_cfg.get('min_degree', 2),
        source_filter=data_cfg.get('source_filter', 'semi'),
        other_possible_fill=data_cfg.get('other_possible_fill', 0.0),
        filter_factset_neg=data_cfg.get('filter_factset_neg', False),
        intra_industry_neg=data_cfg.get('intra_industry_neg', True),
        toy_mode=toy_mode,
    )

    test_dataset = bootstrap_data['test_dataset']
    node_mapping = bootstrap_data['node_mapping']
    reverse_node_mapping = bootstrap_data['reverse_node_mapping']

    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Store (src_graph_idx, tgt_graph_idx, year, label)
    test_records = []
    for link_indices, current_times, labels in tqdm(test_loader, desc="  Extracting test data"):
        for i in range(len(labels)):
            src_idx = int(link_indices[i, 0].item())
            tgt_idx = int(link_indices[i, 1].item())
            year = int(current_times[i].item())
            label = int(labels[i].item())
            test_records.append({
                'src_idx': src_idx,
                'tgt_idx': tgt_idx,
                'year': year,
                'label': label,
            })

    available_years = sorted(bootstrap_data['full_dataset'].available_years)
    return test_records, node_mapping, reverse_node_mapping, available_years


def run_inference(models, test_records, device, batch_size=128, dynamic_data=None):
    """
    Run inference on the test set with all B models.

    Supports two model architectures:
        - GAT-GRU/LSTM/TNA: forward(link_indices, current_times)
        - Temp-SEAL:          forward(dynamic_data, link_indices, current_times)

    Returns:
        prob_matrix: np.ndarray [B, N_test] - prediction probability of each model for each sample
        labels: np.ndarray [N_test] - ground-truth labels
    """
    B = len(models)
    N = len(test_records)

    prob_matrix = np.zeros((B, N))
    labels = np.array([r['label'] for r in test_records])

    node_pairs_list = torch.tensor(
        [[r['src_idx'], r['tgt_idx']] for r in test_records],
        dtype=torch.long
    )
    time_indices_list = torch.tensor(
        [r['year'] for r in test_records], dtype=torch.float
    )

    has_seal = any(hasattr(m, 'subgraph_extractor') for m in models)
    if has_seal and dynamic_data is not None:
        dynamic_data = dynamic_data.to(device)

    for b, model in enumerate(tqdm(models, desc="  Model inference")):
        all_probs = []
        with torch.no_grad():
            for k in range(0, N, batch_size):
                node_pairs = node_pairs_list[k:k + batch_size].to(device)
                time_indices = time_indices_list[k:k + batch_size].to(device)
                if dynamic_data is not None and hasattr(model, 'subgraph_extractor'):
                    preds = model(dynamic_data, node_pairs, time_indices)
                else:
                    preds = model(node_pairs, time_indices)
                if len(preds) > 0:
                    all_probs.append(preds.cpu().numpy().flatten())
        prob_matrix[b] = np.concatenate(all_probs)

    return prob_matrix, labels


# 5.1 Macro performance stability

def compute_macro_stability(prob_matrix, labels):
    """
    Compute macro performance stability metrics for the B models on the test set.

    Returns:
        dict: mean, std, and range of AUC/F1
    """
    B = prob_matrix.shape[0]
    aucs = []
    f1s = []

    for b in range(B):
        probs = prob_matrix[b]
        preds_binary = (probs > 0.5).astype(int)

        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
        else:
            auc = 0.5
        f1 = f1_score(labels, preds_binary, zero_division=0)

        aucs.append(auc)
        f1s.append(f1)

    return {
        'auc': {
            'mean': float(np.mean(aucs)),
            'std': float(np.std(aucs)),
            'min': float(np.min(aucs)),
            'max': float(np.max(aucs)),
            'range': float(np.max(aucs) - np.min(aucs)),
            'values': aucs,
        },
        'f1': {
            'mean': float(np.mean(f1s)),
            'std': float(np.std(f1s)),
            'min': float(np.min(f1s)),
            'max': float(np.max(f1s)),
            'range': float(np.max(f1s) - np.min(f1s)),
            'values': f1s,
        },
    }


# 5.2 Micro individual-prediction stability

def compute_micro_stability(prob_matrix):
    """
    Compute the prediction-probability stability of each sample across the B models.

    Returns:
        dict: per-sample std, and the median and 90th percentile of the std
    """
    # prob_matrix: [B, N] -> std of the B predictions per sample: [N]
    per_sample_std = np.std(prob_matrix, axis=0)

    return {
        'per_sample_std': per_sample_std,
        'median_std': float(np.median(per_sample_std)),
        'p90_std': float(np.percentile(per_sample_std, 90)),
        'p95_std': float(np.percentile(per_sample_std, 95)),
        'mean_std': float(np.mean(per_sample_std)),
        'stable_ratio_005': float(np.mean(per_sample_std < 0.05)),
        'high_variance_ratio_01': float(np.mean(per_sample_std > 0.1)),
    }


# 5.3 Prediction interval width

def compute_prediction_intervals(prob_matrix):
    """
    Compute the 95% prediction interval width for each sample.

    Returns:
        dict: interval width array, mean width, etc.
    """
    # 95% prediction interval: [2.5%, 97.5%]
    p_low = np.percentile(prob_matrix, 2.5, axis=0)
    p_high = np.percentile(prob_matrix, 97.5, axis=0)
    interval_widths = p_high - p_low

    return {
        'interval_widths': interval_widths,
        'mean_width': float(np.mean(interval_widths)),
        'median_width': float(np.median(interval_widths)),
        'p90_width': float(np.percentile(interval_widths, 90)),
        'p_low': p_low,
        'p_high': p_high,
    }


# Visualization

def plot_results(macro_results, micro_results, interval_results, prob_matrix, labels, save_dir):
    """Generate all visualization charts"""
    B = prob_matrix.shape[0]
    
    fig = plt.figure(figsize=(20, 14))
    
    # --- 1. AUC/F1 box plot ---
    ax1 = plt.subplot(3, 3, 1)
    auc_values = macro_results['auc']['values']
    f1_values = macro_results['f1']['values']
    bp = ax1.boxplot([auc_values, f1_values], labels=['AUC', 'F1'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#4ECDC4')
    bp['boxes'][1].set_facecolor('#FF6B6B')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Macro Performance Stability (B={B})\n'
                  f'AUC: {macro_results["auc"]["mean"]:.4f} ± {macro_results["auc"]["std"]:.4f}\n'
                  f'F1:  {macro_results["f1"]["mean"]:.4f} ± {macro_results["f1"]["std"]:.4f}')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # --- 2. AUC per model ---
    ax2 = plt.subplot(3, 3, 2)
    x = range(B)
    ax2.bar(x, auc_values, color='#4ECDC4', edgecolor='white')
    ax2.axhline(y=macro_results['auc']['mean'], color='red', linestyle='--', 
                label=f'Mean={macro_results["auc"]["mean"]:.4f}')
    ax2.axhline(y=macro_results['auc']['mean'] + macro_results['auc']['std'], color='orange', 
                linestyle=':', alpha=0.7, label=f'±1σ')
    ax2.axhline(y=macro_results['auc']['mean'] - macro_results['auc']['std'], color='orange', 
                linestyle=':', alpha=0.7)
    ax2.set_xlabel('Bootstrap Model Index')
    ax2.set_ylabel('AUC')
    ax2.set_title(f'AUC Range: {macro_results["auc"]["range"]:.4f}')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # --- 3. Per-sample std histogram ---
    ax3 = plt.subplot(3, 3, 3)
    per_sample_std = micro_results['per_sample_std']
    ax3.hist(per_sample_std, bins=50, alpha=0.7, edgecolor='black', color='#95E1D3')
    ax3.axvline(micro_results['median_std'], color='red', linestyle='--',
                label=f"Median={micro_results['median_std']:.4f}")
    ax3.axvline(micro_results['p90_std'], color='orange', linestyle='--',
                label=f"P90={micro_results['p90_std']:.4f}")
    ax3.set_xlabel('Prediction Std Dev')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Per-Sample Prediction Stability\n'
                  f'(<0.05: {micro_results["stable_ratio_005"]:.1%}, '
                  f'>0.1: {micro_results["high_variance_ratio_01"]:.1%})')
    ax3.legend(fontsize=8)
    
    # --- 4. CDF of std ---
    ax4 = plt.subplot(3, 3, 4)
    sorted_std = np.sort(per_sample_std)
    cdf = np.arange(1, len(sorted_std) + 1) / len(sorted_std)
    ax4.plot(sorted_std, cdf, linewidth=2)
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    ax4.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90th pct')
    ax4.set_xlabel('Prediction Std Dev')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('CDF of Per-Sample Prediction Std Dev')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    
    # --- 5. 95% prediction interval width histogram ---
    ax5 = plt.subplot(3, 3, 5)
    interval_widths = interval_results['interval_widths']
    ax5.hist(interval_widths, bins=50, alpha=0.7, edgecolor='black', color='#F38181')
    ax5.axvline(interval_results['mean_width'], color='red', linestyle='--',
                label=f"Mean={interval_results['mean_width']:.4f}")
    ax5.set_xlabel('95% Prediction Interval Width')
    ax5.set_ylabel('Count')
    ax5.set_title(f'Prediction Interval Width Distribution\n'
                  f'(Mean Width: {interval_results["mean_width"]:.4f})')
    ax5.legend(fontsize=8)
    
    # --- 6. Interval width CDF ---
    ax6 = plt.subplot(3, 3, 6)
    sorted_width = np.sort(interval_widths)
    cdf_w = np.arange(1, len(sorted_width) + 1) / len(sorted_width)
    ax6.plot(sorted_width, cdf_w, linewidth=2, color='#F38181')
    ax6.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    ax6.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90th pct')
    ax6.set_xlabel('95% Prediction Interval Width')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('CDF of Prediction Interval Width')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    # --- 7. Score distributions by model ---
    ax7 = plt.subplot(3, 3, 7)
    for b in range(B):
        pos_probs = prob_matrix[b][labels == 1]
        neg_probs = prob_matrix[b][labels == 0]
        if len(pos_probs) > 0:
            ax7.hist(pos_probs, bins=30, alpha=0.3, label=f'M{b}' if b < 5 else None)
    ax7.set_xlabel('Prediction Probability (Positive samples)')
    ax7.set_ylabel('Count')
    ax7.set_title('Positive Score Distributions Across Models')
    
    # --- 8. Prediction scatter for first 2 models ---
    ax8 = plt.subplot(3, 3, 8)
    if B >= 2:
        ax8.scatter(prob_matrix[0], prob_matrix[1], alpha=0.3, s=5, 
                    c=['red' if l == 1 else 'blue' for l in labels])
        ax8.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax8.set_xlabel('Model 0 Probability')
        ax8.set_ylabel('Model 1 Probability')
        ax8.set_title(f'Model 0 vs Model 1 Predictions\n'
                      f'(r={np.corrcoef(prob_matrix[0], prob_matrix[1])[0,1]:.4f})')
    
    # --- 9. Summary table ---
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = (
        f"=== Bootstrap Stability Report (B={B}) ===\n\n"
        f"--- Macro Performance ---\n"
        f"AUC: {macro_results['auc']['mean']:.4f} ± {macro_results['auc']['std']:.4f}  "
        f"[{macro_results['auc']['min']:.4f}, {macro_results['auc']['max']:.4f}]\n"
        f"F1:  {macro_results['f1']['mean']:.4f} ± {macro_results['f1']['std']:.4f}  "
        f"[{macro_results['f1']['min']:.4f}, {macro_results['f1']['max']:.4f}]\n\n"
        f"--- Micro Stability ---\n"
        f"Mean Per-Sample Std: {micro_results['mean_std']:.4f}\n"
        f"Median Per-Sample Std: {micro_results['median_std']:.4f}\n"
        f"P90 Per-Sample Std: {micro_results['p90_std']:.4f}\n"
        f"Stable (<0.05) Ratio: {micro_results['stable_ratio_005']:.1%}\n"
        f"High-Var (>0.1) Ratio: {micro_results['high_variance_ratio_01']:.1%}\n\n"
        f"--- Prediction Intervals ---\n"
        f"Mean 95% PI Width: {interval_results['mean_width']:.4f}\n"
        f"Median 95% PI Width: {interval_results['median_width']:.4f}"
    )
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'bootstrap_stability_report.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {save_path}")

    # --- Extra chart: per-sample prediction mean vs std ---
    fig2, ax = plt.subplots(figsize=(10, 6))
    mean_probs = np.mean(prob_matrix, axis=0)
    # Plot negatives (blue) first, then positives (red), so blue does not overpaint red
    neg_mask = labels == 0
    pos_mask = labels == 1
    ax.scatter(mean_probs[neg_mask], per_sample_std[neg_mask],
               alpha=0.3, s=10, c='blue', label='Negative')
    ax.scatter(mean_probs[pos_mask], per_sample_std[pos_mask],
               alpha=0.3, s=10, c='red', label='Positive')
    ax.axhline(0.05, color='green', linestyle='--', label='Stable threshold (0.05)')
    ax.axhline(0.1, color='orange', linestyle='--', label='High-variance threshold (0.1)')
    ax.set_xlabel('Mean Prediction Probability')
    ax.set_ylabel('Prediction Std Dev Across Bootstrap Models')
    ax.set_title(f'Prediction Mean vs Stability (B={B})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_vs_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: mean_vs_stability.png")


# Main entry point

def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap v2 stability analysis'
    )
    parser.add_argument('--dir', '-d', required=True,
                        help='Bootstrap training output directory')
    parser.add_argument('--device', default=None,
                        help='Device (default read from config.yaml)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Inference batch size (default read from config.yaml)')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f"Directory does not exist: {args.dir}")

    # Load config.yaml for default values
    with open(os.path.join(args.dir, 'config.yaml'), 'r') as f:
        run_cfg = yaml.safe_load(f)

    # Device: CLI > config.yaml > auto
    analysis_cfg = run_cfg.get('analysis', {})
    device_str = args.device or analysis_cfg.get('device', 'auto')
    device_str = resolve_device(device_str)
    device = torch.device(device_str)

    # Batch size: CLI > config.yaml > 128
    infer_batch_size = args.batch_size or analysis_cfg.get('batch_size', 128)

    # Note: the model selection score = 0.5 x FactSet quantile + 0.5 x test AUC;
    #       the F1 reported here uses a fixed 0.5 threshold.

    print(f"\n{'='*60}")
    print(f"Bootstrap v2 Stability Analysis")
    print(f"  Directory: {args.dir}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # 1. Extract test data (graph index format)
    print("[Step 1] Extracting test-set data...")
    test_records, node_mapping, reverse_node_mapping, available_years = \
        extract_test_data_with_graph_indices(args.dir, infer_batch_size)
    labels = np.array([r['label'] for r in test_records])
    print(f"  Test set: {len(test_records)} samples (pos: {labels.sum()}, neg: {len(labels)-labels.sum()})")

    # 2. Load dynamic_data
    print("\n[Step 2] Loading static graph...")
    dynamic_data_path = os.path.join(args.dir, 'dynamic_data.pyg')
    dynamic_data = torch.load(dynamic_data_path, weights_only=False)
    print(f"  {dynamic_data.num_nodes} nodes, {dynamic_data.edge_index.shape[1]} edges")

    # 3. Load models
    print("\n[Step 3] Loading Bootstrap models...")

    # Get the model config name from _meta
    meta = run_cfg.get('_meta', {})
    model_config_name = meta.get('model_config_name', run_cfg['model']['config'])

    model_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'Models', 'configs', f"{model_config_name}.yaml"
    )
    if not os.path.exists(model_config_path):
        # Backward compatibility: may be a full path directly
        model_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Models', 'configs', f"gatgru_vec.yaml"
        )
    with open(model_config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ModelClass = import_attr(cfg['model']['module'], cfg['model']['class'])

    num_bootstrap = run_cfg['bootstrap']['num_iterations']
    models = []
    for b in range(num_bootstrap):
        model_path = os.path.join(args.dir, f'best_model_{b}.pth')
        if not os.path.exists(model_path):
            print(f"  Model {b} does not exist: {model_path}")
            continue

        auto_context = {
            'num_features': dynamic_data.x.size(1),
            'num_nodes': dynamic_data.num_nodes,
            'time_steps': available_years,
            'hidden_dims': dynamic_data.x.size(1),
            'device': str(device),
        }
        model_kwargs = resolve_auto_kwargs(cfg['model']['kwargs'], auto_context)

        model_init_params = inspect.signature(ModelClass.__init__).parameters
        if 'dynamic_data' in model_init_params:
            model = ModelClass(dynamic_data=dynamic_data, **model_kwargs)
        else:
            model = ModelClass(**model_kwargs)

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)

    B = len(models)
    print(f"  Successfully loaded {B} models")

    # 4. Inference
    print(f"\n[Step 4] Inference (B={B} models x {len(test_records)} samples)...")
    prob_matrix, labels = run_inference(models, test_records, device, infer_batch_size, dynamic_data)
    print(f"  Inference complete, prob_matrix shape: {prob_matrix.shape}")

    # 5. Compute metrics
    print("\n[Step 5] Computing stability metrics...")

    # 5.1 Macro performance stability
    macro_results = compute_macro_stability(prob_matrix, labels)
    print(f"\n  === 5.1 Macro Performance Stability ===")
    print(f"  AUC: {macro_results['auc']['mean']:.4f} +/- {macro_results['auc']['std']:.4f}  "
          f"range: [{macro_results['auc']['min']:.4f}, {macro_results['auc']['max']:.4f}]  "
          f"range width: {macro_results['auc']['range']:.4f}")
    print(f"  F1:  {macro_results['f1']['mean']:.4f} +/- {macro_results['f1']['std']:.4f}  "
          f"range: [{macro_results['f1']['min']:.4f}, {macro_results['f1']['max']:.4f}]  "
          f"range width: {macro_results['f1']['range']:.4f}")

    # 5.2 Micro individual-prediction stability
    micro_results = compute_micro_stability(prob_matrix)
    print(f"\n  === 5.2 Micro Individual-Prediction Stability ===")
    print(f"  Mean std: {micro_results['mean_std']:.4f}")
    print(f"  Median std: {micro_results['median_std']:.4f}")
    print(f"  90th percentile std: {micro_results['p90_std']:.4f}")
    print(f"  95th percentile std: {micro_results['p95_std']:.4f}")
    print(f"  Stable sample (<0.05) ratio: {micro_results['stable_ratio_005']:.1%}")
    print(f"  High-variance sample (>0.1) ratio: {micro_results['high_variance_ratio_01']:.1%}")

    # 5.3 Prediction interval width
    interval_results = compute_prediction_intervals(prob_matrix)
    print(f"\n  === 5.3 Prediction Interval Width ===")
    print(f"  Mean 95% prediction interval width: {interval_results['mean_width']:.4f}")
    print(f"  Median 95% prediction interval width: {interval_results['median_width']:.4f}")
    print(f"  90th percentile width: {interval_results['p90_width']:.4f}")

    # 6. Visualization
    print("\n[Step 6] Generating visualizations...")
    plot_results(macro_results, micro_results, interval_results, prob_matrix, labels, args.dir)

    # 7. Save results
    print("\n[Step 7] Saving results...")

    # Prediction probability matrix
    prob_df = pd.DataFrame(
        prob_matrix.T,
        columns=[f'model_{i}_prob' for i in range(B)]
    )
    prob_df['label'] = labels
    prob_df['mean_prob'] = np.mean(prob_matrix, axis=0)
    prob_df['std_prob'] = micro_results['per_sample_std']
    prob_df['pi_low'] = interval_results['p_low']
    prob_df['pi_high'] = interval_results['p_high']
    prob_df['pi_width'] = interval_results['interval_widths']

    # Original triple information
    prob_df['src_idx'] = [r['src_idx'] for r in test_records]
    prob_df['tgt_idx'] = [r['tgt_idx'] for r in test_records]
    prob_df['year'] = [r['year'] for r in test_records]
    prob_df.to_csv(os.path.join(args.dir, 'prediction_matrix.csv'), index=False)

    # Summary metrics
    summary = {
        'num_models': B,
        'num_test_samples': len(test_records),
        'macro_auc_mean': macro_results['auc']['mean'],
        'macro_auc_std': macro_results['auc']['std'],
        'macro_auc_range': macro_results['auc']['range'],
        'macro_f1_mean': macro_results['f1']['mean'],
        'macro_f1_std': macro_results['f1']['std'],
        'macro_f1_range': macro_results['f1']['range'],
        'micro_mean_std': micro_results['mean_std'],
        'micro_median_std': micro_results['median_std'],
        'micro_p90_std': micro_results['p90_std'],
        'micro_p95_std': micro_results['p95_std'],
        'micro_stable_ratio_005': micro_results['stable_ratio_005'],
        'micro_high_variance_ratio_01': micro_results['high_variance_ratio_01'],
        'interval_mean_width': interval_results['mean_width'],
        'interval_median_width': interval_results['median_width'],
        'interval_p90_width': interval_results['p90_width'],
    }

    summary_path = os.path.join(args.dir, 'stability_summary.yaml')
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f)

    print(f"\nBootstrap v2 analysis complete!")
    print(f"   Summary metrics: {summary_path}")
    print(f"   Prediction matrix: prediction_matrix.csv")
    print(f"   Visualizations: bootstrap_stability_report.png")
    print(f"                   mean_vs_stability.png")


if __name__ == "__main__":
    main()
