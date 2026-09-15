# -*- coding: utf-8 -*-
"""
PrROC2_fff: Draw a single-model ROC curve from GAT-GRU[fff] test-set predictions.
Data source: model_predictions_best_auc.npy of some fff-config run (test-set predictions
        of the best-AUC checkpoint). By default the latest one under <results-root>/gatgru/*fff*/
        is selected automatically; the run directory can also be specified via the
        IMPUT_FFF_RUN environment variable (relative to the repo root or absolute), and the
        results root via IMPUT_RESULTS_DIR (needed when results/ is foldered, e.g. results/单次4模式).
Output: Analysis/npy_roc_output/fig_roc_gatgru_fff.png
      Analysis/npy_roc_output/roc_curve_gatgru_fff.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
# Results root: IMPUT_RESULTS_DIR (absolute or relative to the repo root) or <repo>/results
_RESULTS_ROOT = os.environ.get('IMPUT_RESULTS_DIR') or os.path.join(_PROJECT_ROOT, 'results')
if not os.path.isabs(_RESULTS_ROOT):
    _RESULTS_ROOT = os.path.join(_PROJECT_ROOT, _RESULTS_ROOT)

def _resolve_npy_file():
    run = os.environ.get('IMPUT_FFF_RUN')
    if run:
        base = run if os.path.isabs(run) else os.path.join(_PROJECT_ROOT, run)
        return os.path.join(base, 'model_predictions_best_auc.npy')
    import glob
    pattern = os.path.join(_RESULTS_ROOT, 'gatgru', '*fff*',
                           'model_predictions_best_auc.npy')
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"Prediction file for the fff config not found: {pattern}\n"
            "Please run Training/train_gatgru.py or SampleSetting/run_sampling.py first, "
            "or specify the run directory via the IMPUT_FFF_RUN environment variable.")
    return candidates[-1]


NPY_FILE = _resolve_npy_file()
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'npy_roc_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE_CSV = os.path.join(_SCRIPT_DIR, 'edgebank_baseline_results.csv')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

MODEL_LABEL = 'GAT-GRU [fff]'
MODEL_COLOR = '#1f77b4'


def main():
    data = np.load(NPY_FILE, allow_pickle=True).item()
    pos = np.asarray(data['test_predictions'][0]).flatten()
    neg = np.asarray(data['neg_predictions'][0]).flatten()

    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Youden best operating point
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_fpr, best_tpr = fpr[best_idx], tpr[best_idx]
    best_thresh = thresholds[best_idx]
    n_pos, n_neg = len(pos), len(neg)
    tp = np.sum(y_score[:n_pos] >= best_thresh)
    fp = np.sum(y_score[n_pos:] >= best_thresh)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_pos if n_pos > 0 else 0.0

    print(f"Samples: pos={n_pos}, neg={n_neg}")
    print(f"AUC = {roc_auc:.4f}")
    print(f"Youden best point: FPR={best_fpr:.4f}, TPR={best_tpr:.4f}, "
          f"Prec={precision:.4f}, Rec={recall:.4f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5,
            label='Random Guess (AUC=0.5)')
    ax.plot(fpr, tpr, linewidth=3, alpha=0.9, color=MODEL_COLOR,
            label=f'{MODEL_LABEL} (AUC={roc_auc:.3f})')
    ax.scatter([best_fpr], [best_tpr], s=150, color=MODEL_COLOR,
               edgecolors='white', linewidth=2.5, zorder=5, marker='o')
    ax.annotate(f'{MODEL_LABEL}\nPrec={precision:.3f}, Rec={recall:.3f}',
                xy=(best_fpr, best_tpr),
                xytext=(best_fpr + 0.06, best_tpr - 0.10),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=MODEL_COLOR, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

    # EdgeBank baseline points (W=window shape, T=threshold color)
    if os.path.exists(BASELINE_CSV):
        baseline_df = pd.read_csv(BASELINE_CSV)
        window_shapes = {2: 'o', 4: 's', 8: '^', 16: 'D'}
        threshold_colors_map = {1: '#d7191c', 2: '#fdae61', 3: '#2b83ba'}
        for _, row in baseline_df.iterrows():
            shape = window_shapes.get(int(row['window']), 'o')
            color = threshold_colors_map.get(int(row['threshold']), '#7f7f7f')
            ax.scatter(row['fpr'], row['tpr'], s=80, color=color, marker=shape,
                       edgecolors='white', linewidth=1, zorder=6, alpha=0.6,
                       label='_nolegend_')
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        baseline_legend_elements = []
        for window, shape in window_shapes.items():
            baseline_legend_elements.append(
                Line2D([0], [0], marker=shape, color='w', markerfacecolor='gray',
                       markersize=8, label=f'W={window}',
                       markeredgecolor='k', markeredgewidth=0.5))
        for threshold in sorted(threshold_colors_map):
            color = threshold_colors_map[threshold]
            baseline_legend_elements.append(
                Patch(facecolor=color, edgecolor='k', alpha=0.6,
                      label=f'T={threshold}'))
        baseline_legend = ax.legend(
            handles=baseline_legend_elements, title='EdgeBank\n(W=Window, T=Threshold)',
            loc='center right', fontsize=9, framealpha=0.9, title_fontsize=9)
        ax.add_artist(baseline_legend)

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    # Title is provided by the paper's LaTeX caption; no title is drawn in the figure
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, 'fig_roc_gatgru_fff.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC curve saved: '{png_path}'")

    csv_path = os.path.join(OUTPUT_DIR, 'roc_curve_gatgru_fff.csv')
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(csv_path, index=False)
    print(f"Curve data saved: '{csv_path}'")
    print(f"Best point: FPR={best_fpr:.4f}, TPR={best_tpr:.4f}, "
          f"Precision={precision:.4f}, Recall={recall:.4f}")


if __name__ == '__main__':
    main()
