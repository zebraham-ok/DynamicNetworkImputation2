#!/usr/bin/env python3
"""
Seed-ensemble ROC figures with 95% confidence intervals.

Where Analysis/seed_ci_summary.py works on the per-seed summary.yaml values, this script goes one
level deeper: it reads the saved test-set scores of every repeat and turns them into ROC curves
whose spread across seeds is shown explicitly.

Input layout (the "ensemble" layout produced for the four-repeat runs, i.e. the unpacked archives):

    <results-dir>/<model>/<run>/seed<NN>/<i>_<flag>_<role>/model_predictions_best_auc.npy

Each npy holds the test-set scores of the best-val-AUC checkpoint,
`{'test_predictions': [pos_scores], 'neg_predictions': [neg_scores]}`.

Outputs (default Analysis/npy_roc_output_ci/, next to the single-run Analysis/npy_roc_output/):

    fig_roc_ci_grid_points.png     per-seed ROC + mean curve with a 95% CI band, one panel per
                                   (model, flag), plus the mean Youden / F1-max points
    fig_roc_ci_overlay_points.png  one panel per flag, mean ROC of every model with its 95% CI band
                                   and its mean operating points (thresholds in the legend)
    fig_roc_ci_overlay_<flag>_points_zoom.png
                                   one flag per figure: left = full range, right = the operating
                                   region zoomed in with the numeric thresholds
    roc_ci_summary.csv      per (model, flag): AUC mean / std / 95% CI / per-seed AUCs
    roc_ci_summary.md       the same table in markdown
    roc_ci_curves.csv       mean lower/upper TPR on a common FPR grid (for re-plotting elsewhere)

Usage:
    python Analysis/seed_ci_figures.py [--results-dir DIR] [--out-dir DIR]
                                       [--models bigru egcn ...] [--flags fff ftf]

Which models appear and how they are named comes from `model_plot_config.json` next to the results
root (see Analysis/plot_models.py); `--models` still overrides it when given.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import plot_models  # noqa: E402  (Analysis/plot_models.py, stdlib-only)

FLAG_BY_STAGE = {'_ftt_': 'ftt', '_ftf_': 'ftf', '_fff_': 'fff', '_tff_': 'tff'}
FLAG_ORDER = ['ftt', 'ftf', 'fff', 'tff']
N_FPR = 201
FPR_GRID = np.linspace(0.0, 1.0, N_FPR)

# Display names: reuse the single source of truth in visualize_results.py, with a local fallback
# so the script still runs if it is copied next to a different analysis tree.
_FALLBACK_LABELS = {'bigru': 'BiGRU', 'egcn': 'EvolveGCN-H', 'egcn-smooth': 'EvolveGCN-H-LS',
                    'gatgru': 'GAT-GRU', 'gatgru2layer': 'GAT-GRU*',
                    'gatgru2layer-smooth': 'GAT-GRU-LS', 'tna': 'BiTNA',
                    'seal': 'SEAL', 'fusion1layer': 'GAT-GRU-FiLM', 'fusion2layer': 'GAT-GRU*-FiLM'}
try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from visualize_results import MODEL_DISPLAY as _MODEL_DISPLAY
    MODEL_LABELS = dict(_FALLBACK_LABELS)
    MODEL_LABELS.update({k: v for k, v in _MODEL_DISPLAY.items() if v})
except Exception:  # pragma: no cover - only when visualize_results is unavailable
    MODEL_LABELS = dict(_FALLBACK_LABELS)

from seed_ci_summary import _t95  # noqa: E402  (same directory; identical small-sample convention)
from PrROC_all_models_points import (  # noqa: E402  (same convention as the run directories'
    _fmt_thr, _place_threshold_labels, _recorded_thresholds)  # *_thresholds.json files)
from matplotlib.lines import Line2D  # noqa: E402

MODEL_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728', '#8c564b', '#17becf',
                '#bcbd22', '#9edae5']

# EdgeBank baseline overlay — same CSV and same visual convention as PrROC.py / PrROC2_fff.py /
# PrROC_all_models.py: marker shape = window length W, colour = decision threshold T.
BASELINE_CSV = SCRIPT_DIR / 'edgebank_baseline_results.csv'
WINDOW_SHAPES = {2: 'o', 4: 's', 8: '^', 16: 'D'}
THRESHOLD_COLORS = {1: '#d7191c', 2: '#fdae61', 3: '#2b83ba'}
EDGEBANK_LEGEND_TITLE = 'EdgeBank\n(W=Window, T=Threshold)'


def label_of(model):
    return MODEL_LABELS.get(model, model)


def _trapezoid(y, x):
    """np.trapz was renamed np.trapezoid in NumPy 2.0 (the old alias is deprecated)."""
    fn = getattr(np, 'trapezoid', None) or np.trapz
    return float(fn(y, x))


def flag_of(stage_name):
    for key, flag in FLAG_BY_STAGE.items():
        if key in stage_name:
            return flag
    return None


def _seed_of(dirname):
    m = re.fullmatch(r'seed(\d+)', dirname)
    return int(m.group(1)) if m else None


def discover(results_dir):
    """Walk the ensemble tree -> {(model, flag): [(seed, npy_path), ...]} preserving seed order."""
    found = {}
    for model_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                seed = _seed_of(seed_dir.name)
                if seed is None:
                    continue
                for stage_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                    flag = flag_of(stage_dir.name)
                    if flag is None:
                        continue
                    npy = stage_dir / 'model_predictions_best_auc.npy'
                    if not npy.exists():
                        print(f"  [WARN] missing {npy}")
                        continue
                    found.setdefault((model_dir.name, flag), []).append((seed, npy))
    for key in found:
        found[key].sort()
    return found


def load_scores(npy_path):
    data = np.load(str(npy_path), allow_pickle=True).item()
    pos = np.asarray(data['test_predictions'][0]).flatten()
    neg = np.asarray(data['neg_predictions'][0]).flatten()
    return pos, neg


def roc_auc(pos, neg):
    """ROC on a common FPR grid + trapezoidal AUC (no sklearn dependency needed, kept explicit)."""
    from sklearn.metrics import roc_curve, auc
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, auc(fpr, tpr)


def build_curves(entries):
    """Per-seed ROC + AUC, plus the mean curve and its 95% CI band on FPR_GRID.

    Each seed also carries its Youden / F1-max operating point (same convention as the run
    directories' *_thresholds.json); their mean is reported as `ops` and drawn on the figures.
    """
    per_seed = []
    for seed, npy in entries:
        pos, neg = load_scores(npy)
        fpr, tpr, auc_val = roc_auc(pos, neg)
        thr = _recorded_thresholds(pos, neg)
        per_seed.append({
            'seed': seed,
            'fpr': fpr,
            'tpr': tpr,
            'auc': float(auc_val),
            'n_pos': int(pos.size),
            'n_neg': int(neg.size),
            'youden_threshold': thr['youden_threshold'],
            'f1_threshold': thr['plotted_f1_threshold'],
            'f1_scan_degenerate': bool(thr['f1_scan_degenerate']),
            'f1_scan_mismatch': bool(thr['f1_scan_mismatch']),
            'youden_fpr': thr['youden_fpr'], 'youden_tpr': thr['youden_tpr'],
            'f1_fpr': thr['plotted_f1_fpr'], 'f1_tpr': thr['plotted_f1_tpr'],
            'f1_at_youden': thr['f1_at_youden'], 'f1_at_f1': thr['f1_at_plotted'],
        })

    aucs = np.array([e['auc'] for e in per_seed], dtype=float)
    n = len(aucs)
    mean_auc = float(aucs.mean())
    std_auc = float(aucs.std(ddof=1)) if n > 1 else float('nan')
    ci_auc = float(_t95(n) * std_auc / np.sqrt(n)) if (n > 1 and np.isfinite(std_auc)) else float('nan')

    interp = np.vstack([np.interp(FPR_GRID, e['fpr'], e['tpr']) for e in per_seed])
    mean_tpr = interp.mean(axis=0)
    if n > 1:
        band = _t95(n) * interp.std(axis=0, ddof=1) / np.sqrt(n)
    else:
        band = np.zeros_like(mean_tpr)

    def _avg(key):
        return float(np.mean([e[key] for e in per_seed]))

    def _spread(key):
        vals = [e[key] for e in per_seed]
        return float(min(vals)), float(max(vals))

    def _mean_std_ci(key):
        """Mean ± the same small-sample 95% CI used for the AUC (`t95(n) * std / sqrt(n)`).

        Thresholds are per-seed numbers just like the AUC, so they get the same treatment; the
        spread can be huge (the F1-max scan is noisy), which is exactly what the CI has to show.
        """
        vals = np.array([e[key] for e in per_seed], dtype=float)
        mean_v = float(vals.mean())
        if n > 1:
            std_v = float(vals.std(ddof=1))
            ci_v = float(_t95(n) * std_v / np.sqrt(n))
        else:
            std_v, ci_v = float('nan'), float('nan')
        return mean_v, std_v, ci_v

    youden_mean, youden_std, youden_ci = _mean_std_ci('youden_threshold')
    f1_mean, f1_std, f1_ci = _mean_std_ci('f1_threshold')

    ops = {
        'youden_threshold': youden_mean,
        'youden_threshold_std': youden_std,
        'youden_threshold_ci95': youden_ci,
        'youden_fpr': _avg('youden_fpr'),
        'youden_tpr': _avg('youden_tpr'),
        'f1_at_youden': _avg('f1_at_youden'),
        'f1_threshold': f1_mean,
        'f1_threshold_std': f1_std,
        'f1_threshold_ci95': f1_ci,
        'f1_fpr': _avg('f1_fpr'),
        'f1_tpr': _avg('f1_tpr'),
        'f1_at_f1': _avg('f1_at_f1'),
        'youden_threshold_lo': _spread('youden_threshold')[0],
        'youden_threshold_hi': _spread('youden_threshold')[1],
        'f1_threshold_lo': _spread('f1_threshold')[0],
        'f1_threshold_hi': _spread('f1_threshold')[1],
        'n_seeds_scan_flagged': int(sum(bool(e['f1_scan_degenerate']) or bool(e['f1_scan_mismatch'])
                                        for e in per_seed)),
    }

    return {
        'per_seed': per_seed,
        'n': n,
        'auc_mean': mean_auc,
        'auc_std': std_auc,
        'auc_ci95': ci_auc,
        'auc_min': float(aucs.min()),
        'auc_max': float(aucs.max()),
        'aucs': aucs,
        'mean_tpr': mean_tpr,
        'lo_tpr': np.clip(mean_tpr - band, 0, 1),
        'hi_tpr': np.clip(mean_tpr + band, 0, 1),
        'mean_auc_from_curve': _trapezoid(mean_tpr, FPR_GRID),
        'ops': ops,
    }


def _fmt_auc(curve):
    if curve['n'] > 1 and np.isfinite(curve['auc_ci95']):
        return f"{curve['auc_mean']:.4f} ± {curve['auc_ci95']:.4f}"
    return f"{curve['auc_mean']:.4f} (n=1)"


def _draw_op_points(ax, ops, color):
    """Youden (o) and F1-max (*) operating points of one curve, on top of its CI band."""
    ax.scatter([ops['youden_fpr']], [ops['youden_tpr']], s=95, color=color, marker='o',
               edgecolors='white', linewidth=1.6, zorder=8)
    ax.scatter([ops['f1_fpr']], [ops['f1_tpr']], s=200, color=color, marker='*',
               edgecolors='white', linewidth=1.4, zorder=8)


def load_edgebank(csv_path=BASELINE_CSV):
    """EdgeBank (window, threshold) operating points, as recorded for PrROC.py.

    Returns one dict per configuration: {window, threshold, fpr, tpr, auc}. An empty list when the
    CSV is missing, so the figures still render (just without the baseline overlay).
    """
    import pandas as pd

    if not Path(csv_path).exists():
        print(f"  [WARN] EdgeBank baseline CSV not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    points = [{'window': int(r['window']), 'threshold': int(r['threshold']),
               'fpr': float(r['fpr']), 'tpr': float(r['tpr']), 'auc': float(r['auc'])}
              for _, r in df.iterrows()]
    best = max(points, key=lambda p: p['auc'])
    print(f"  EdgeBank baseline: {len(points)} configs; best W{best['window']}-T{best['threshold']} "
          f"AUC {best['auc']:.4f} at FPR {best['fpr']:.4f} / TPR {best['tpr']:.4f}")
    return points


def _draw_edgebank(ax, points):
    """Scatter the EdgeBank configs of `points`; return the (W, T) legend handles."""
    from matplotlib.patches import Patch

    for p in points:
        ax.scatter([p['fpr']], [p['tpr']], s=70,
                   color=THRESHOLD_COLORS.get(p['threshold'], '#7f7f7f'),
                   marker=WINDOW_SHAPES.get(p['window'], 'o'),
                   edgecolors='white', linewidth=0.8, zorder=3, alpha=0.9,
                   label='_nolegend_')
    handles = [Line2D([0], [0], marker=shape, color='w', markerfacecolor='gray', markersize=8,
                      label=f'W={window}', markeredgecolor='k', markeredgewidth=0.5)
               for window, shape in WINDOW_SHAPES.items()]
    handles += [Patch(facecolor=THRESHOLD_COLORS[t], edgecolor='k', alpha=0.6, label=f'T={t}')
                for t in sorted(THRESHOLD_COLORS)]
    return handles


def _draw_aligned_models_legend(ax, curves, models, flag, fontsize=9.0, anchor=(0.985, 0.025)):
    """Panel-(a) model legend whose two columns are aligned: names left, "AUC ± CI" right.

    `ax.legend` renders each row as one string, so the AUC text starts right after the model name and
    the numbers of different rows never line up. Here a row is split into two text artists — the names
    are left-aligned on a shared edge, the AUC strings right-aligned on another — and the column widths
    come from the rendered glyph extents, so the frame hugs its content for any set of labels.
    Returns the number of rows drawn (0 when the mode has no data).
    """
    from matplotlib.patches import Rectangle

    rows = []
    for m_idx, model in enumerate(models):
        cur = curves.get((model, flag))
        if cur is None:
            continue
        rows.append((MODEL_COLORS[m_idx % len(MODEL_COLORS)], label_of(model),
                     f"AUC {_fmt_auc(cur)}"))
    if not rows:
        return 0

    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    def text_size(s):
        probe = ax.text(0.5, 0.5, s, fontsize=fontsize, transform=ax.transAxes, alpha=0)
        box = probe.get_window_extent(renderer=renderer)
        probe.remove()
        return box.width, box.height

    w_name = max(text_size(r[1])[0] for r in rows)
    w_val = max(text_size(r[2])[0] for r in rows)
    h_row = max(text_size(r[1])[1] for r in rows)

    # All lengths in display pixels, then converted to axes fractions (the axes size is what the
    # legend is laid out against, so the frame keeps its proportions at any dpi).
    aw, ah = ax.bbox.width, ax.bbox.height
    pad_x, pad_y, handle_w, gap_hn, col_gap = 7.0, 6.0, 24.0, 7.0, 12.0
    pitch = h_row * 1.45
    w_box = 2 * pad_x + handle_w + gap_hn + w_name + col_gap + w_val
    h_box = 2 * pad_y + len(rows) * pitch

    x1, y1 = anchor[0], anchor[1] + h_box / ah
    x0 = x1 - w_box / aw
    ax.add_patch(Rectangle((x0, anchor[1]), w_box / aw, h_box / ah, transform=ax.transAxes,
                           facecolor='white', edgecolor='0.75', lw=0.8, alpha=0.9, zorder=4))
    x_h = x0 + pad_x / aw
    x_n = x0 + (pad_x + handle_w + gap_hn) / aw
    x_v = x1 - pad_x / aw
    for i, (color, name, val) in enumerate(rows):
        yc = y1 - (pad_y + (i + 0.5) * pitch) / ah
        ax.add_line(Line2D([x_h, x_h + handle_w / aw], [yc, yc], transform=ax.transAxes, color=color,
                           lw=2.0, zorder=5, clip_on=False, solid_capstyle='round'))
        ax.text(x_n, yc, name, transform=ax.transAxes, ha='left', va='center', fontsize=fontsize,
                zorder=5)
        ax.text(x_v, yc, val, transform=ax.transAxes, ha='right', va='center', fontsize=fontsize,
                zorder=5)
    return len(rows)


def plot_grid(curves, models, flags, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(models), len(flags)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.9 * n_rows), squeeze=False)
    fig.suptitle('Test-set ROC of the best-validation checkpoint — mean ± 95% CI over repeated runs\n'
                 'operating points: o Youden J (max TPR-FPR), * F1-max, both averaged over the seeds',
                 fontsize=14, fontweight='bold', y=0.998)

    for r, model in enumerate(models):
        for c, flag in enumerate(flags):
            ax = axes[r][c]
            key = (model, flag)
            ax.plot([0, 1], [0, 1], color='0.75', lw=0.8, ls='--', zorder=1)
            if key not in curves:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', color='0.5',
                        transform=ax.transAxes)
            else:
                cur = curves[key]
                for e in cur['per_seed']:
                    ax.plot(e['fpr'], e['tpr'], color='0.65', lw=0.8, alpha=0.85, zorder=2,
                            label=f"seed {e['seed']}" if r == 0 and c == 0 else None)
                ax.fill_between(FPR_GRID, cur['lo_tpr'], cur['hi_tpr'], color='#d62728',
                                alpha=0.20, lw=0, zorder=3)
                ax.plot(FPR_GRID, cur['mean_tpr'], color='#d62728', lw=2.0, zorder=4)
                _draw_op_points(ax, cur['ops'], '#d62728')
                ax.text(0.96, 0.06, f"mean AUC {_fmt_auc(cur)}\n{cur['n']} seeds\n"
                                    f"J t={_fmt_thr(cur['ops']['youden_threshold'])} "
                                    f"(F1 {cur['ops']['f1_at_youden']:.3f})\n"
                                    f"F1* t={_fmt_thr(cur['ops']['f1_threshold'])} "
                                    f"(F1 {cur['ops']['f1_at_f1']:.3f})",
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.9))
            if r == 0:
                ax.set_title(flag, fontsize=12, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f"{label_of(model)}\nTrue positive rate", fontsize=10)
            else:
                ax.set_ylabel('True positive rate', fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('False positive rate', fontsize=9)
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _draw_mean_panel(ax, curves, models, flag, with_legend=False, label_mode='full'):
    """Mean curve + CI band + operating points of every model of one negative-sampling mode.

    label_mode='full' writes the AUC and both operating thresholds into the line label;
    label_mode='auc' keeps only "name: AUC ± CI" (the zoom figure prints the thresholds itself).
    """
    ax.plot([0, 1], [0, 1], color='0.75', lw=0.9, ls='--')
    items = []
    for m_idx, model in enumerate(models):
        key = (model, flag)
        if key not in curves:
            continue
        cur = curves[key]
        color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]
        ops = cur['ops']
        ax.fill_between(FPR_GRID, cur['lo_tpr'], cur['hi_tpr'], color=color, alpha=0.15, lw=0)
        label = f"{label_of(model)}: AUC {_fmt_auc(cur)}"
        if label_mode == 'full':
            label += (f" | J t={_fmt_thr(ops['youden_threshold'])}"
                      f" | F1* t={_fmt_thr(ops['f1_threshold'])}")
        ax.plot(FPR_GRID, cur['mean_tpr'], color=color, lw=2.0, label=label)
        _draw_op_points(ax, ops, color)
        items.append((((ops['youden_fpr']), (ops['youden_tpr'])),
                      f"J {_fmt_thr(ops['youden_threshold'])}", color, 'o'))
        items.append((((ops['f1_fpr']), (ops['f1_tpr'])),
                      f"F1* {_fmt_thr(ops['f1_threshold'])}", color, '*'))
    ax.set_xlabel('False positive rate', fontsize=10)
    ax.set_ylabel('True positive rate', fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    if with_legend:
        ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    return items


def plot_overlay(curves, models, flags, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cols = len(flags)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 5.0), squeeze=False)
    fig.suptitle('Mean test ROC over repeated runs (95% CI band), by negative-sampling mode\n'
                 'markers: o Youden J (max TPR-FPR), * F1-max; the legend gives both thresholds',
                 fontsize=13, fontweight='bold', y=0.99)
    for c, flag in enumerate(flags):
        ax = axes[0][c]
        _draw_mean_panel(ax, curves, models, flag, with_legend=True)
        ax.set_title(flag, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overlay_zoom(curves, models, flag, path, edgebank=None):
    """One mode per figure, left = full range, right = the operating region zoomed in.

    Mirrors the layout of `Analysis/PrROC_all_models_points.py` so the two ROC families read the
    same way: the right panel carries the numeric thresholds of every operating point. The EdgeBank
    configurations (same CSV / same W-T convention as PrROC.py) go on the left panel only — the
    right panel stays a pure model-operating-region zoom, so the baseline points never force the
    window down into the near-trivial corner — and the left legend is kept to "model: AUC ± CI"
    because the thresholds are written on the right panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    edgebank = edgebank or []
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(16.5, 8.2))
    fig.suptitle(f'Mean test ROC over repeated runs, mode {flag} — 95% CI band, the averaged '
                 f'Youden / F1-max operating points, and the EdgeBank baseline (panel a)',
                 fontsize=14, fontweight='bold', y=0.995)

    _draw_mean_panel(ax, curves, models, flag, label_mode='auc')
    ax.set_title('(a) full range', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    if edgebank:
        base_handles = _draw_edgebank(ax, edgebank)
        # Inside panel (a), not below it: the area right of x~0.3 below the curves is empty at this
        # scale, whereas an outside legend would collide with the (a) x-label and the tight bbox.
        # ncol=2 with the handles ordered W-first then T gives two vertical stacks: left = windows,
        # right = thresholds (matplotlib fills legend columns in consecutive chunks of the handle list).
        legend_base = ax.legend(handles=base_handles, title=EDGEBANK_LEGEND_TITLE,
                                loc='center right', bbox_to_anchor=(1.0, 0.45),
                                ncol=2, columnspacing=1.8, fontsize=8,
                                framealpha=0.9, title_fontsize=8.5)
        ax.add_artist(legend_base)
    # Model legend is drawn by hand (two aligned columns) instead of ax.legend(loc='lower right'):
    # the names start on a shared left edge and the AUC ± CI values end on a shared right edge.
    _draw_aligned_models_legend(ax, curves, models, flag, fontsize=9)

    items = _draw_mean_panel(axz, curves, models, flag)
    axz.set_title('(b) zoom on the operating region', fontsize=12, fontweight='bold')

    # zoom window derived from the operating points of this mode, so nothing can fall outside
    xy = []
    for model in models:
        cur = curves.get((model, flag))
        if cur is None:
            continue
        xy += [(cur['ops']['youden_fpr'], cur['ops']['youden_tpr']),
               (cur['ops']['f1_fpr'], cur['ops']['f1_tpr'])]
    if xy:
        x_hi = max(0.05, min(0.5, max(p[0] for p in xy) * 1.15 + 0.005))
        y_lo = max(0.0, min(0.95, min(p[1] for p in xy) - 0.07))
        y_hi = min(1.005, max(p[1] for p in xy) + 0.07)
    else:  # pragma: no cover - only when the mode has no predictions at all
        x_hi, y_lo, y_hi = 0.15, 0.6, 1.0
    # Deliberately derived from the model operating points only: EdgeBank lives on panel (a), so panel
    # (b) must not be widened down into the near-trivial corner just to contain the baseline points.
    axz.set_xlim(-x_hi * 0.06, x_hi)
    axz.set_ylim(y_lo, y_hi)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.45', markersize=9,
               markeredgecolor='white', label='Youden J: argmax(TPR - FPR)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='0.45', markersize=15,
               markeredgecolor='white', label='F1-max: argmax F1'),
    ]
    legend_m = axz.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.95,
                          title='operating points (mean over seeds)', title_fontsize=9.5)
    axz.add_artist(legend_m)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reserved = [legend_m.get_window_extent(renderer=renderer)]
    _place_threshold_labels(axz, items, reserved)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def write_tables(curves, models, flags, csv_path, md_path, curve_csv_path):
    import pandas as pd

    rows = []
    for model in models:
        for flag in flags:
            cur = curves.get((model, flag))
            if cur is None:
                continue
            rows.append({
                'model': model,
                'display': label_of(model),
                'flag': flag,
                'n_seeds': cur['n'],
                'seeds': ', '.join(str(e['seed']) for e in cur['per_seed']),
                'auc_mean': cur['auc_mean'],
                'auc_std': cur['auc_std'],
                'auc_ci95': cur['auc_ci95'],
                'auc_min': cur['auc_min'],
                'auc_max': cur['auc_max'],
                'auc_mean_from_mean_curve': cur['mean_auc_from_curve'],
                'per_seed_auc': ', '.join(f"{e['seed']}:{e['auc']:.4f}" for e in cur['per_seed']),
                'youden_threshold_mean': cur['ops']['youden_threshold'],
                'youden_threshold_std': cur['ops']['youden_threshold_std'],
                'youden_threshold_ci95': cur['ops']['youden_threshold_ci95'],
                'youden_threshold_min': cur['ops']['youden_threshold_lo'],
                'youden_threshold_max': cur['ops']['youden_threshold_hi'],
                'per_seed_youden': ', '.join(f"{e['seed']}:{e['youden_threshold']:.3f}"
                                             for e in cur['per_seed']),
                'f1_threshold_mean': cur['ops']['f1_threshold'],
                'f1_threshold_std': cur['ops']['f1_threshold_std'],
                'f1_threshold_ci95': cur['ops']['f1_threshold_ci95'],
                'f1_threshold_min': cur['ops']['f1_threshold_lo'],
                'f1_threshold_max': cur['ops']['f1_threshold_hi'],
                'per_seed_f1max': ', '.join(f"{e['seed']}:{e['f1_threshold']:.3f}"
                                            for e in cur['per_seed']),
                'f1_at_youden_mean': cur['ops']['f1_at_youden'],
                'f1_at_f1max_mean': cur['ops']['f1_at_f1'],
                'seeds_with_offrange_f1_scan': cur['ops']['n_seeds_scan_flagged'],
                'n_pos': cur['per_seed'][0]['n_pos'],
                'n_neg': cur['per_seed'][0]['n_neg'],
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {csv_path}")

    lines = ['# Seed-ensemble ROC / AUC (test set of the best-validation checkpoint)', '',
             'Per-seed AUCs are averaged over the repeats; the 95% CI is the two-sided small-sample '
             'CI (`t95(n) * std / sqrt(n)`), i.e. the same convention as `seed_ci_summary.py`.', '']
    if summary.empty:
        lines.append('_No prediction files found._')
    else:
        lines.append('| Model | Mode | Mean AUC ± 95% CI | Min – Max | Youden thr (mean ± 95% CI) | '
                     'F1-max thr (mean ± 95% CI) | Seeds | Per-seed AUC |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for _idx, row in summary.iterrows():
            lines.append(f"| {row['display']} | {row['flag']} | "
                         f"{row['auc_mean']:.4f} ± {row['auc_ci95']:.4f} | "
                         f"{row['auc_min']:.4f} – {row['auc_max']:.4f} | "
                         f"{_fmt_thr(row['youden_threshold_mean'])} ± "
                         f"{_fmt_thr(row['youden_threshold_ci95'])} | "
                         f"{_fmt_thr(row['f1_threshold_mean'])} ± "
                         f"{_fmt_thr(row['f1_threshold_ci95'])} | "
                         f"{row['seeds']} | {row['per_seed_auc']} |")
        lines.append('')
        lines.append('Operating points: Youden J = argmax(TPR - FPR); F1-max = the grid scan '
                     '`np.arange(0.1, 0.9, 0.001)` of `utils.youden_f1max_thresholds` (exact argmax '
                     'over the observed scores when that scan is off-range). Thresholds are '
                     'averaged over the seeds with the same small-sample 95% CI as the AUC; the '
                     'per-seed values, the min–max range and the F1 at each point are in the CSV. '
                     'A wide threshold CI is expected when the seeds disagree (the F1-max scan in '
                     'particular can sit at very different scores), so the CI is the uncertainty of '
                     'the mean, not a bound on the per-seed operating points.')
        lines.append('')
    Path(md_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Saved: {md_path}")

    # mean curve table, long format, so the bands can be redrawn without re-reading the npy files
    curve_rows = []
    for (model, flag), cur in sorted(curves.items()):
        for i, fpr in enumerate(FPR_GRID):
            curve_rows.append({'model': model, 'flag': flag, 'fpr': fpr,
                               'tpr_mean': cur['mean_tpr'][i], 'tpr_lo': cur['lo_tpr'][i],
                               'tpr_hi': cur['hi_tpr'][i]})
    pd.DataFrame(curve_rows).to_csv(curve_csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {curve_csv_path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir',
                    default=os.environ.get('IMPUT_RESULTS_DIR')
                    or str(REPO_ROOT / 'results' / '四次双模式' / 'ensemble'),
                    help='root holding <model>/<run>/seed<NN>/<i>_<flag>_<role>/model_predictions_best_auc.npy')
    ap.add_argument('--out-dir', default=os.environ.get('IMPUT_OUT_DIR')
                    or str(SCRIPT_DIR / 'npy_roc_output_ci'))
    ap.add_argument('--models', nargs='+', default=None, help='restrict to these model directories')
    ap.add_argument('--model-config', default=None,
                    help='plot list overriding which models are drawn / how they are named '
                         f'(default: {plot_models.CONFIG_NAME} next to the results root)')
    ap.add_argument('--flags', nargs='+', default=None, help='restrict to these flags')
    ap.add_argument('--auc-from', choices=['mean_of_seeds', 'mean_curve'], default='mean_of_seeds',
                    help='reported AUC: average of the per-seed AUCs (default) or the area of the '
                         'mean ROC curve (the latter is slightly optimistic under averaging)')
    ap.add_argument('--baseline-csv', default=str(BASELINE_CSV),
                    help='EdgeBank baseline configurations (default: the CSV used by PrROC.py)')
    ap.add_argument('--no-baseline', action='store_true',
                    help='skip the EdgeBank overlay of the two zoom figures')
    return ap.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    print('Seed-ensemble ROC with confidence intervals')
    print(f'  results : {results_dir}')
    print(f'  output  : {out_dir}')
    if not results_dir.is_dir():
        print(f'  [ERROR] not a directory: {results_dir}')
        return

    found = discover(results_dir)
    if not found:
        print('  [ERROR] no model_predictions_best_auc.npy found — is this the ensemble layout?')
        return

    # The plot list decides which models are drawn and how they are labelled; its order is also the
    # colour order below, so commenting a line drops the model (and its colour slot) everywhere.
    cfg = plot_models.load_plot_models(results_dir, args.model_config)
    allowed = None
    if cfg is not None:
        allowed = cfg.mapping(available={m for m, _ in found}, base_names=MODEL_LABELS)
        MODEL_LABELS.update(allowed)
        found = {k: v for k, v in found.items() if k[0] in allowed}
        print(f"  plot list    : {cfg.path.name} — {len(allowed)} models "
              f"({', '.join(MODEL_LABELS.get(m, m) for m in allowed)})")
        if not found:
            print('  [ERROR] the plot config selects no model present in this tree')
            return

    if args.models:
        models = list(args.models)
    elif allowed:
        models = [m for m in allowed if any(k[0] == m for k in found)]
    else:
        models = list(dict.fromkeys(m for m, _ in found))
    flags = [f for f in (args.flags or FLAG_ORDER) if any(k[1] == f for k in found)]

    curves = {}
    for key, entries in found.items():
        model, flag = key
        if model not in models or flag not in flags:
            continue
        curves[key] = build_curves(entries)
        print(f"  {label_of(model):<16} {flag}  n={curves[key]['n']}  "
              f"AUC={_fmt_auc(curves[key])}  per-seed={np.round(curves[key]['aucs'], 4).tolist()}")

    if not curves:
        print('  [ERROR] nothing to plot after filtering')
        return

    if args.auc_from == 'mean_curve':
        for cur in curves.values():
            n = cur['n']
            std = float(np.std([e['auc'] for e in cur['per_seed']], ddof=1)) if n > 1 else float('nan')
            cur['auc_mean'] = cur['mean_auc_from_curve']
            cur['auc_std'] = std
            cur['auc_ci95'] = float(_t95(n) * std / np.sqrt(n)) if n > 1 else float('nan')

    out_dir.mkdir(parents=True, exist_ok=True)
    edgebank = [] if args.no_baseline else load_edgebank(args.baseline_csv)
    plot_grid(curves, models, flags, out_dir / 'fig_roc_ci_grid_points.png')
    plot_overlay(curves, models, flags, out_dir / 'fig_roc_ci_overlay_points.png')
    for flag in flags:
        plot_overlay_zoom(curves, models, flag,
                          out_dir / f'fig_roc_ci_overlay_{flag}_points_zoom.png',
                          edgebank=edgebank)
    write_tables(curves, models, flags, out_dir / 'roc_ci_summary.csv',
                 out_dir / 'roc_ci_summary.md', out_dir / 'roc_ci_curves.csv')
    print(f'\n  Done. Output directory: {out_dir}')


if __name__ == '__main__':
    main()
