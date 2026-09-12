# -*- coding: utf-8 -*-
"""
PrROC_all_models: one ROC curve per model, each taken from that model's best-AUC run.

Companion of Analysis/PrROC2_fff.py (single model, fixed fff config). This script sweeps
every model directory under results/ and, for each model, keeps the negative-sampling
configuration (ftt/ftf/fff/tff) with the highest test AUC, so the figure answers the
question "how high does each architecture get on its own best setting".

Data source: results/<model>/<MMDD-HHMM>-<flag>/model_predictions_best_auc.npy
    {'test_predictions': [pos_scores (1, N_pos)], 'neg_predictions': [neg_scores (1, N_neg)]}
    of the best-AUC checkpoint, evaluated on the held-out test split.

Outputs (Analysis/npy_roc_output/):
    fig_roc_all_models.png          ROC curves of all models (best config each) + EdgeBank
    roc_curves_all_models.csv       long format: model, flag, run_dir, auc, fpr, tpr
    roc_best_auc_summary.csv        one row per model: AUC / Youden point / Precision / Recall / F1

Usage:
    python Analysis/PrROC_all_models.py
    python Analysis/PrROC_all_models.py --flag fff          # same config for every model
    python Analysis/PrROC_all_models.py --models gatgru seal --out-dir Analysis/npy_roc_output
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Display names follow Analysis/visualize_results.py::MODEL_DISPLAY; the two-layer GAT-GRU
# (results/gatgru2layer) and the one-directional variant are written GAT-GRU* / GAT-GRU†.
MODEL_DISPLAY = {
    'bigru': 'BiGRU',
    'egcn': 'EvolveGCN-H',
    'gatgru': 'GAT-GRU',
    'gatgru2layer': 'GAT-GRU*',
    'seal': 'SEAL',
    'tna': 'BiTNA',
}

# GAT-GRU keeps the colour of PrROC2_fff.py so the two figures can be placed side by side.
MODEL_COLORS = {
    'gatgru': '#1f77b4',
    'gatgru2layer': '#17becf',
    'bigru': '#2ca02c',
    'egcn': '#d62728',
    'tna': '#8c564b',
    'seal': '#9467bd',
}

BASELINE_CSV = os.path.join(_SCRIPT_DIR, 'edgebank_baseline_results.csv')
WINDOW_SHAPES = {2: 'o', 4: 's', 8: '^', 16: 'D'}
THRESHOLD_COLORS = {1: '#d7191c', 2: '#fdae61', 3: '#2b83ba'}

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description='ROC curves of all models, each on its best-AUC negative-sampling config')
    parser.add_argument('--results-dir', default=os.path.join(_PROJECT_ROOT, 'results'),
                        help='root holding <model>/<run>/model_predictions_best_auc.npy '
                             '(default: <repo>/results)')
    parser.add_argument('--out-dir', default=os.path.join(_SCRIPT_DIR, 'npy_roc_output'),
                        help='output directory (default: Analysis/npy_roc_output)')
    parser.add_argument('--flag', default=None, choices=['ftt', 'ftf', 'fff', 'tff'],
                        help='force one negative-sampling config for every model '
                             '(default: per-model best AUC)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='restrict to these model directories (default: all found)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='skip the EdgeBank baseline points')
    return parser.parse_args()


def discover_runs(results_dir, models=None):
    """{model: [{flag, run_dir, npy_path}, ...]} sorted by flag."""
    runs = {}
    for npy_path in sorted(glob.glob(os.path.join(results_dir, '*', '*',
                                                  'model_predictions_best_auc.npy'))):
        run_dir = os.path.dirname(npy_path)
        model = os.path.basename(os.path.dirname(run_dir))
        if models and model not in models:
            continue
        run_name = os.path.basename(run_dir)
        runs.setdefault(model, []).append({
            'flag': run_name.rsplit('-', 1)[-1],
            'run_dir': run_name,
            'npy_path': npy_path,
        })
    return {m: sorted(v, key=lambda r: r['flag']) for m, v in runs.items()}


def load_curve(npy_path):
    """Positive/negative test scores of the best-AUC checkpoint -> (fpr, tpr, thresholds)."""
    data = np.load(npy_path, allow_pickle=True).item()
    pos = np.asarray(data['test_predictions'][0]).flatten()
    neg = np.asarray(data['neg_predictions'][0]).flatten()
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds, pos, neg


def _metrics_at(y_score, n_pos, thresh):
    """Precision / recall / F1 at a given score threshold (positives first in y_score)."""
    tp = np.sum(y_score[:n_pos] >= thresh)
    fp = np.sum(y_score[n_pos:] >= thresh)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_pos if n_pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def build_records(runs, forced_flag):
    """One record per model: the run with the highest test AUC (or the forced config)."""
    records, all_curves = [], []
    for model in sorted(runs):
        if model not in MODEL_DISPLAY:
            print(f"[WARN] no display name for '{model}' - the directory name is used")
        candidates = []
        for run in runs[model]:
            if forced_flag and run['flag'] != forced_flag:
                continue
            fpr, tpr, thresholds, pos, neg = load_curve(run['npy_path'])
            y_score = np.concatenate([pos, neg])
            idx = int(np.argmax(tpr - fpr))
            precision, recall, f1 = _metrics_at(y_score, len(pos), thresholds[idx])
            candidates.append({
                'model': model,
                'display': MODEL_DISPLAY.get(model, model),
                'flag': run['flag'],
                'run_dir': f"{model}/{run['run_dir']}",
                'n_pos': int(len(pos)),
                'n_neg': int(len(neg)),
                'auc': float(auc(fpr, tpr)),
                'youden_fpr': float(fpr[idx]),
                'youden_tpr': float(tpr[idx]),
                'youden_threshold': float(thresholds[idx]),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr,
                'tpr': tpr,
            })
        if not candidates:
            print(f"[WARN] model '{model}' has no run matching flag={forced_flag}; skipped")
            continue
        best = max(candidates, key=lambda r: r['auc'])
        records.append(best)
        for cand in candidates:
            all_curves.append({
                'model': cand['display'], 'flag': cand['flag'], 'run_dir': cand['run_dir'],
                'auc': cand['auc'], 'selected': cand is best,
                'fpr': cand['fpr'], 'tpr': cand['tpr'],
            })
    records.sort(key=lambda r: r['auc'], reverse=True)
    return records, all_curves


def _place_labels(ax, records, colors, reserved_bboxes):
    """Annotate the Youden points with the model names, pushing labels into free space.

    adjustText is not a project dependency, so a small deterministic search over candidate
    offsets is used: the first offset whose text box clears the already placed labels is taken.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    candidates = [(14, 14), (14, -14), (14, -42), (-14, 14), (-14, -14),
                  (14, 42), (-14, 42), (-14, -42), (46, 6), (-46, 6), (6, 56), (6, -56)]
    occupied = list(reserved_bboxes)
    for rec in records:
        color = colors.get(rec['model'], '#7f7f7f')
        for dx, dy in candidates:
            ha = 'left' if dx >= 0 else 'right'
            va = 'bottom' if dy >= 0 else 'top'
            ann = ax.annotate(rec['display'], xy=(rec['youden_fpr'], rec['youden_tpr']),
                              xytext=(dx, dy), textcoords='offset points',
                              fontsize=10, fontweight='bold', color=color, ha=ha, va=va,
                              bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                                        edgecolor=color, alpha=0.85, linewidth=0.8),
                              arrowprops=dict(arrowstyle='-', color=color, lw=0.6, alpha=0.8),
                              zorder=8)
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
            if not any(bbox.overlaps(prev) for prev in occupied):
                occupied.append(bbox)
                break
            ann.remove()
        else:
            # every candidate collided: keep the first one (already removed) at its default spot
            ann = ax.annotate(rec['display'], xy=(rec['youden_fpr'], rec['youden_tpr']),
                              xytext=(14, 14), textcoords='offset points',
                              fontsize=10, fontweight='bold', color=color,
                              bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                                        edgecolor=color, alpha=0.85, linewidth=0.8),
                              arrowprops=dict(arrowstyle='-', color=color, lw=0.6),
                              zorder=8)
            fig.canvas.draw()
            occupied.append(ann.get_window_extent(renderer=renderer))


def plot_curves(records, out_png, baseline_csv=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5,
            label='Random Guess (AUC=0.5)')

    for rec in records:
        color = MODEL_COLORS.get(rec['model'], '#7f7f7f')
        ax.plot(rec['fpr'], rec['tpr'], linewidth=2.6, alpha=0.9, color=color,
                label=f"{rec['display']} [{rec['flag']}] (AUC={rec['auc']:.3f})")
        ax.scatter([rec['youden_fpr']], [rec['youden_tpr']], s=110, color=color,
                   edgecolors='white', linewidth=2.0, zorder=7, marker='o')

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    reserved = []
    # EdgeBank baseline (W = window shape, T = threshold colour), as in PrROC2_fff.py
    if baseline_csv and os.path.exists(baseline_csv):
        baseline_df = pd.read_csv(baseline_csv)
        for _, row in baseline_df.iterrows():
            ax.scatter(row['fpr'], row['tpr'],
                       s=80, color=THRESHOLD_COLORS.get(int(row['threshold']), '#7f7f7f'),
                       marker=WINDOW_SHAPES.get(int(row['window']), 'o'),
                       edgecolors='white', linewidth=1, zorder=6, alpha=0.6)
        handles = [Line2D([0], [0], marker=shape, color='w', markerfacecolor='gray',
                          markersize=8, label=f'W={window}', markeredgecolor='k',
                          markeredgewidth=0.5)
                   for window, shape in WINDOW_SHAPES.items()]
        handles += [Patch(facecolor=THRESHOLD_COLORS[t], edgecolor='k', alpha=0.6,
                          label=f'T={t}') for t in sorted(THRESHOLD_COLORS)]
        legend_base = ax.legend(handles=handles, title='EdgeBank\n(W=Window, T=Threshold)',
                                loc='center right', fontsize=9, framealpha=0.9,
                                title_fontsize=9)
        ax.add_artist(legend_base)
        fig.canvas.draw()
        reserved.append(legend_base.get_window_extent(fig.canvas.get_renderer()))

    legend = ax.legend(loc='lower right', fontsize=11, framealpha=0.9,
                       title='Model [config] - test AUC', title_fontsize=11)
    fig.canvas.draw()
    reserved.append(legend.get_window_extent(fig.canvas.get_renderer()))

    _place_labels(ax, records, MODEL_COLORS, reserved)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC figure saved: '{out_png}'")


def save_tables(records, all_curves, out_dir):
    curve_rows = []
    for curve in all_curves:
        curve_rows.append(pd.DataFrame({
            'model': curve['model'], 'flag': curve['flag'], 'run_dir': curve['run_dir'],
            'auc': curve['auc'], 'selected': curve['selected'],
            'fpr': curve['fpr'], 'tpr': curve['tpr'],
        }))
    curves_path = os.path.join(out_dir, 'roc_curves_all_models.csv')
    pd.concat(curve_rows, ignore_index=True).to_csv(curves_path, index=False)
    print(f"Curve data saved: '{curves_path}'")

    summary_cols = ['display', 'flag', 'run_dir', 'n_pos', 'n_neg', 'auc',
                    'youden_fpr', 'youden_tpr', 'youden_threshold',
                    'precision', 'recall', 'f1']
    summary = pd.DataFrame([{k: r[k] for k in summary_cols} for r in records])
    summary = summary.rename(columns={'display': 'model'})
    summary_path = os.path.join(out_dir, 'roc_best_auc_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Best-AUC summary saved: '{summary_path}'")
    return summary


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 60)
    mode = f"forced config: {args.flag}" if args.flag else 'per-model best AUC'
    print(f"PrROC_all_models: ROC of every model ({mode})")
    print(f"results : {args.results_dir}")
    print(f"output  : {args.out_dir}")
    print('=' * 60)

    runs = discover_runs(args.results_dir, args.models)
    if not runs:
        raise FileNotFoundError(
            f"No model_predictions_best_auc.npy found under {args.results_dir}/<model>/<run>/; "
            "run SampleSetting/run_sampling.py first.")
    for model, model_runs in sorted(runs.items()):
        flags = ', '.join(r['flag'] for r in model_runs)
        print(f"  {model:<14} {len(model_runs)} run(s): {flags}")

    records, all_curves = build_records(runs, args.flag)
    if not records:
        raise RuntimeError('No usable run after filtering.')

    print('-' * 78)
    print(f"{'Model':<16} {'cfg':<5} {'AUC':>8} {'FPR@J':>8} {'TPR@J':>8} "
          f"{'Prec':>8} {'Rec':>8} {'F1':>8}")
    print('-' * 78)
    for rec in records:
        print(f"{rec['display']:<16} {rec['flag']:<5} {rec['auc']:>8.4f} "
              f"{rec['youden_fpr']:>8.4f} {rec['youden_tpr']:>8.4f} "
              f"{rec['precision']:>8.4f} {rec['recall']:>8.4f} {rec['f1']:>8.4f}")
    print('-' * 78)
    n_pos = records[0]['n_pos']
    n_neg = records[0]['n_neg']
    print(f"Samples per model: pos={n_pos}, neg={n_neg} (1:1 test split, "
          f"{n_pos + n_neg} pairs)")

    baseline = None if args.no_baseline else BASELINE_CSV
    plot_curves(records, os.path.join(args.out_dir, 'fig_roc_all_models.png'), baseline)
    save_tables(records, all_curves, args.out_dir)


if __name__ == '__main__':
    main()
