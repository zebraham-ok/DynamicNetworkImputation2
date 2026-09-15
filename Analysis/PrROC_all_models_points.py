# -*- coding: utf-8 -*-
"""
PrROC_all_models_points: same ROC panel as Analysis/PrROC_all_models.py, but with the two
operating points of every curve drawn explicitly and their thresholds written on the figure:

    circle (o)   Youden's J      argmax(TPR - FPR), threshold read off the roc_curve grid
    star   (*)   F1-max          argmax F1 over the grid scan np.arange(0.1, 0.9, 0.001)

Both conventions are copied verbatim from utils.youden_f1max_thresholds (the function that writes each
run's `model_predictions_best_auc_thresholds.json`), so the numbers printed here are the same ones
the run directories report. utils.py itself cannot be imported from a plotting environment because it
imports torch at module level, hence the local re-implementation - keep the two in sync.

Everything else (which run is used per model, the colour map, the EdgeBank baseline overlay) is
inherited from PrROC_all_models.py by import, so this script only adds the operating points.

Data source: <results-dir>/<model>/<run>/model_predictions_best_auc.npy, as in PrROC_all_models.py.

Outputs (Analysis/npy_roc_output/):
    fig_roc_all_models_f1youden.png    all models with the Youden / F1-max points of every curve
    roc_f1_youden_summary.csv          per model: AUC, both thresholds, both operating points, F1

The F1-max grid scan is half-open (0.9 excluded) and starts at 0.1, so for a model whose scores do
not live in [0.1, 0.9) - SEAL saturates near 0 - the scan is degenerate. Such a row is reported with
`f1_scan_degenerate = True`, the exact argmax over the observed scores is added as a reference, and
the figure marks the exact point with a hollow star instead of pretending the scan succeeded.

Usage:
    python Analysis/PrROC_all_models_points.py
    python Analysis/PrROC_all_models_points.py --flag fff --out-name fig_roc_all_models_fff_f1youden.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import f1_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from PrROC_all_models import (  # noqa: E402  (same directory, single source of truth)
    BASELINE_CSV, MODEL_COLORS, MODEL_DISPLAY, THRESHOLD_COLORS, WINDOW_SHAPES,
    _place_labels, build_records, discover_runs, load_curve,
)
import plot_models  # noqa: E402  (same directory; stdlib-only)

# Identical to utils.F1_SCAN (Prediction/find_threshold*.py convention). Half-open interval.
F1_SCAN = (0.1, 0.9, 0.001)

# F1 difference above which the grid scan is treated as having missed the real optimum
F1_SCAN_MISMATCH_TOL = 0.005


def _fmt_thr(t):
    """Threshold label: three decimals in the usual range, scientific notation at the extremes."""
    if t is None:
        return 'n/a'
    if 5e-4 <= abs(t) <= 0.9995:
        return f"{t:.3f}"
    return f"{t:.1e}"


def _recorded_thresholds(pos, neg, f1_scan=F1_SCAN):
    """Youden / F1-max thresholds for one score pair, matching utils.youden_f1max_thresholds."""
    from sklearn.metrics import roc_curve

    y_true = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
    y_score = np.concatenate([pos, neg])

    out = {'n_pos': int(len(pos)), 'n_neg': int(len(neg))}

    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_j = int(np.argmax(j_scores))
    thr_j = float(roc_thr[best_j])
    out['youden_threshold'] = thr_j
    out['youden_fpr'] = float(fpr[best_j])
    out['youden_tpr'] = float(tpr[best_j])
    out['youden_j_stat'] = float(j_scores[best_j])
    out['youden_at_inf'] = not np.isfinite(thr_j)
    if out['youden_at_inf']:            # constant scores: the ROC grid puts argmax J on +inf
        out['youden_threshold'] = float(np.max(y_score))
        out['youden_fpr'] = float(np.mean(y_score[~y_true.astype(bool)] >= out['youden_threshold']))
        out['youden_tpr'] = float(np.mean(y_score[y_true.astype(bool)] >= out['youden_threshold']))
    out['f1_at_youden'] = float(
        f1_score(y_true, (y_score >= out['youden_threshold']).astype(int), zero_division=0))

    # canonical grid scan
    start, stop, step = f1_scan
    best_f1, best_t = 0.0, None
    for t in np.arange(start, stop, step):
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    out['f1_scan'] = [float(start), float(stop), float(step)]
    out['f1_scan_degenerate'] = best_t is None
    out['f1max_threshold'] = float(start) if best_t is None else best_t
    out['f1_at_f1max'] = best_f1

    # exact argmax F1 over the observed scores, as a reference (and the plotted point when the
    # canonical scan is degenerate). Sweeping the sorted scores gives, after k items,
    # TP = tp[k], FP = fp[k] and FN = n_pos - tp[k], hence F1 = 2 TP / (TP + FP + n_pos).
    order = np.argsort(-y_score, kind='mergesort')
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_pos, n_neg = float(tp[-1]), float(fp[-1])
    f1_exact = 2.0 * tp / (tp + fp + n_pos)
    best_e = int(np.argmax(f1_exact))
    thr_e = float(y_score[order][best_e])
    out['f1max_exact_threshold'] = thr_e
    out['f1_at_f1max_exact'] = float(f1_exact[best_e])
    out['f1max_exact_fpr'] = float(fp[best_e] / n_neg) if n_neg > 0 else 0.0
    out['f1max_exact_tpr'] = float(tp[best_e] / n_pos) if n_pos > 0 else 0.0

    # The canonical scan only looks at [0.1, 0.9). When the scores live outside that window the scan
    # either finds nothing (degenerate) or stops at its edge, far below the true optimum - SEAL, whose
    # saturating scores sit near 0, is the standing example. In both cases the exact argmax over the
    # observed scores is plotted instead and flagged, rather than quoting a scan artefact.
    out['f1_scan_mismatch'] = bool(
        (out['f1_at_f1max_exact'] - out['f1_at_f1max']) > F1_SCAN_MISMATCH_TOL)

    # the F1-max point that is drawn: canonical scan when usable, exact argmax otherwise
    if out['f1_scan_degenerate'] or out['f1_scan_mismatch']:
        out['plotted_f1_threshold'] = thr_e
        out['plotted_f1_fpr'] = out['f1max_exact_fpr']
        out['plotted_f1_tpr'] = out['f1max_exact_tpr']
        out['plotted_f1_source'] = ('exact (scan degenerate)' if out['f1_scan_degenerate']
                                    else 'exact (scan missed optimum)')
    else:
        t = out['f1max_threshold']
        out['plotted_f1_threshold'] = t
        out['plotted_f1_fpr'] = float(np.mean(neg >= t))
        out['plotted_f1_tpr'] = float(np.mean(pos >= t))
        out['plotted_f1_source'] = 'scan 0.1-0.9'

    # distance between the two operating points, to see how far apart the criteria sit
    out['delta_threshold'] = out['plotted_f1_threshold'] - out['youden_threshold']
    out['f1_at_plotted'] = float(
        f1_score(y_true, (y_score >= out['plotted_f1_threshold']).astype(int), zero_division=0))
    return out


def enrich_records(runs, forced_flag, results_dir):
    """build_records + the Youden / F1-max operating points of every selected run."""
    records, all_curves = build_records(runs, forced_flag)
    for rec in records:
        model, run_name = rec['run_dir'].split('/')
        npy = next(r['npy_path'] for r in runs[model] if r['run_dir'] == run_name)
        _fpr, _tpr, _thr, pos, neg = load_curve(npy)
        rec.update(_recorded_thresholds(pos, neg))
    return records, all_curves


def _place_threshold_labels(ax, items, reserved):
    """Small collision-avoided numeric labels (the threshold values) next to each marker."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = [(10, 5), (-10, 5), (10, -13), (-10, -13), (10, 15), (-10, 15),
                  (24, 0), (-24, 0), (0, 11), (0, -19), (10, 26), (-10, 26),
                  (34, 9), (-34, -9), (0, 27), (0, -31)]
    ax_box = ax.get_window_extent(renderer=renderer)
    occupied = list(reserved)
    for xy, text, color, marker in items:
        for dx, dy in candidates:
            ann = ax.annotate(text, xy=xy, xytext=(dx, dy), textcoords='offset points',
                              fontsize=8.5, color=color, ha='left' if dx >= 0 else 'right',
                              va='bottom' if dy >= 0 else 'top', zorder=9,
                              bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                        edgecolor=color, alpha=0.85, linewidth=0.6))
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.06, 1.16)
            inside = (bbox.x0 >= ax_box.x0 - 2 and bbox.x1 <= ax_box.x1 + 2
                      and bbox.y0 >= ax_box.y0 - 2 and bbox.y1 <= ax_box.y1 + 2)
            if inside and not any(bbox.overlaps(prev) for prev in occupied):
                occupied.append(bbox)
                break
            ann.remove()
        else:
            ann = ax.annotate(text, xy=xy, xytext=(9, 6), textcoords='offset points',
                              fontsize=8.5, color=color, ha='left', va='bottom', zorder=9,
                              bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                        edgecolor=color, alpha=0.85, linewidth=0.6))
            fig.canvas.draw()
            occupied.append(ann.get_window_extent(renderer=renderer))


def _draw_panel(ax, records, with_curve_labels):
    """Curves + the two operating points of every model on one axes."""
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Guess (AUC=0.5)')
    for rec in records:
        color = MODEL_COLORS.get(rec['model'], '#7f7f7f')
        ax.plot(rec['fpr'], rec['tpr'], linewidth=2.4, alpha=0.9, color=color,
                label=(f"{rec['display']} [{rec['flag']}]  AUC={rec['auc']:.3f}  "
                       f"J t={_fmt_thr(rec['youden_threshold'])}  "
                       f"F1* t={_fmt_thr(rec['plotted_f1_threshold'])}")
                if with_curve_labels else None)
        ax.scatter([rec['youden_fpr']], [rec['youden_tpr']], s=120, color=color,
                   edgecolors='white', linewidth=2.0, zorder=7, marker='o')
        star_kwargs = dict(s=260, color=color, edgecolors='white', linewidth=1.8, zorder=7,
                           marker='*')
        if rec['f1_scan_degenerate'] or rec['f1_scan_mismatch']:
            star_kwargs.update(facecolors='none', edgecolors=color, linewidth=1.8)
        ax.scatter([rec['plotted_f1_fpr']], [rec['plotted_f1_tpr']], **star_kwargs)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')


def plot_curves_points(records, out_png, baseline_csv=None, subtitle=None):
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(17.5, 8.6))

    _draw_panel(ax, records, with_curve_labels=True)
    ax.set_title('(a) full range', fontsize=12, fontweight='bold')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    _draw_panel(axz, records, with_curve_labels=False)
    axz.set_title('(b) zoom on the operating region', fontsize=12, fontweight='bold')

    reserved = []
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

    legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.9,
                       title='Model [config] \u2014 test AUC and the two thresholds '
                             '(J: Youden, F1*: F1-max)', title_fontsize=10)
    fig.canvas.draw()
    reserved.append(legend.get_window_extent(fig.canvas.get_renderer()))

    # zoom limits: every operating point must be visible, with a little head-room
    ops_x = [r['youden_fpr'] for r in records] + [r['plotted_f1_fpr'] for r in records]
    ops_y = [r['youden_tpr'] for r in records] + [r['plotted_f1_tpr'] for r in records]
    x_hi = float(np.clip(max(ops_x) * 1.25 + 0.005, 0.05, 0.5))
    y_lo = float(np.clip(min(ops_y) - 0.12, 0.0, 0.95))
    axz.set_xlim(-x_hi * 0.06, x_hi)
    axz.set_ylim(y_lo, 1.005)

    # marker semantics, in the free lower-left corner of the zoom panel
    marker_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.45', markersize=10,
               markeredgecolor='white', label='Youden J: argmax(TPR - FPR)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='0.45', markersize=16,
               markeredgecolor='white', label='F1-max, 0.1-0.9 threshold scan'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='none', markersize=16,
               markeredgecolor='0.45', markeredgewidth=1.6,
               label='F1-max outside that scan:\nexact argmax over the scores'),
    ]
    legend_m = axz.legend(handles=marker_handles, loc='lower left', fontsize=9, framealpha=0.95,
                          title='operating points', title_fontsize=9.5)
    axz.add_artist(legend_m)
    fig.canvas.draw()
    reserved.append(legend_m.get_window_extent(fig.canvas.get_renderer()))

    # model names at the Youden points (reused from PrROC_all_models), then the threshold values
    _place_labels(axz, records, MODEL_COLORS, reserved)
    fig.canvas.draw()
    for rec in records:
        an = [a for a in axz.texts if a.get_text() == rec['display']]
        if an:
            reserved.append(an[-1].get_window_extent(fig.canvas.get_renderer()))

    items = []
    for rec in records:
        color = MODEL_COLORS.get(rec['model'], '#7f7f7f')
        if abs(rec['delta_threshold']) < 1e-9:
            # the two criteria select the same score: one combined label, drawn once
            items.append(((rec['youden_fpr'], rec['youden_tpr']),
                          f"J = F1* {_fmt_thr(rec['youden_threshold'])}", color, 'o'))
            continue
        items.append((((rec['plotted_f1_fpr']), (rec['plotted_f1_tpr'])),
                      f"F1* {_fmt_thr(rec['plotted_f1_threshold'])}", color, '*'))
        items.append(((rec['youden_fpr'], rec['youden_tpr']),
                      f"J {_fmt_thr(rec['youden_threshold'])}", color, 'o'))
    _place_threshold_labels(axz, items, reserved)

    fig.suptitle(subtitle or 'ROC of each model on its best-AUC configuration', fontsize=14,
                 fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC figure with operating points saved: '{out_png}'")


def save_summary(records, out_csv):
    cols = ['display', 'flag', 'run_dir', 'n_pos', 'n_neg', 'auc',
            'youden_threshold', 'youden_fpr', 'youden_tpr', 'youden_j_stat', 'f1_at_youden',
            'f1max_threshold', 'plotted_f1_threshold', 'plotted_f1_fpr', 'plotted_f1_tpr',
            'f1_at_f1max', 'f1_at_plotted', 'f1_scan_degenerate', 'f1_scan_mismatch',
            'plotted_f1_source',
            'f1max_exact_threshold', 'f1_at_f1max_exact', 'delta_threshold']
    summary = pd.DataFrame([{k: r[k] for k in cols} for r in records])
    summary = summary.rename(columns={'display': 'model'})
    summary.to_csv(out_csv, index=False)
    print(f"Operating-point summary saved: '{out_csv}'")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='ROC of all models with the Youden / F1-max operating points and thresholds')
    default_results = os.path.join(_PROJECT_ROOT, 'results')
    single = os.path.join(default_results, '单次4模式')
    parser.add_argument('--results-dir', default=single if os.path.isdir(single) else default_results,
                        help='root holding <model>/<run>/model_predictions_best_auc.npy')
    parser.add_argument('--out-dir', default=os.path.join(_SCRIPT_DIR, 'npy_roc_output'))
    parser.add_argument('--out-name', default='fig_roc_all_models_f1youden.png')
    parser.add_argument('--summary-name', default='roc_f1_youden_summary.csv')
    parser.add_argument('--flag', default=None, choices=['ftt', 'ftf', 'fff', 'tff'],
                        help='force one negative-sampling config for every model '
                             '(default: per-model best AUC)')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--model-config', default=None,
                        help='plot list deciding which models are drawn / how they are named '
                             f'(default: {plot_models.CONFIG_NAME} next to the results root)')
    parser.add_argument('--no-baseline', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 78)
    print(f"PrROC_all_models_points: ROC + Youden / F1-max operating points "
          f"({f'forced config: {args.flag}' if args.flag else 'per-model best AUC'})")
    print(f"results : {args.results_dir}")
    print(f"output  : {args.out_dir}")
    print('=' * 78)

    # The plot list decides which models appear and how they are labelled; --models still wins.
    cfg = plot_models.load_plot_models(args.results_dir, args.model_config)
    if cfg is not None:
        available = {d for d in os.listdir(args.results_dir)
                     if os.path.isdir(os.path.join(args.results_dir, d))} \
            if os.path.isdir(args.results_dir) else set()
        names = cfg.mapping(available=available, base_names=MODEL_DISPLAY)
        MODEL_DISPLAY.update(names)
        if not args.models:
            args.models = list(names)

    runs = discover_runs(args.results_dir, args.models)
    if not runs:
        raise FileNotFoundError(f"No model_predictions_best_auc.npy found under {args.results_dir}")

    records, _all_curves = enrich_records(runs, args.flag, args.results_dir)
    if not records:
        raise RuntimeError('No usable run after filtering.')

    print('-' * 112)
    print(f"{'Model':<16} {'cfg':<5} {'AUC':>7} {'J thr':>10} {'J F1':>7} {'F1* thr':>10} "
          f"{'F1* F1':>7} {'d thr':>8}  {'F1* source':<26}")
    print('-' * 112)
    for rec in records:
        print(f"{rec['display']:<16} {rec['flag']:<5} {rec['auc']:>7.4f} "
              f"{_fmt_thr(rec['youden_threshold']):>10} {rec['f1_at_youden']:>7.4f} "
              f"{_fmt_thr(rec['plotted_f1_threshold']):>10} {rec['f1_at_plotted']:>7.4f} "
              f"{rec['delta_threshold']:>8.4f}  {rec['plotted_f1_source']:<26}")
    print('-' * 112)
    if any(r['f1_scan_degenerate'] or r['f1_scan_mismatch'] for r in records):
        print("[NOTE] for the flagged model(s) the 0.1-0.9 grid scan of utils.youden_f1max_thresholds")
        print("       cannot reach the real maximum (score range outside the grid): the exact argmax")
        print("       over the observed scores is drawn as a hollow star instead.")

    baseline = None if args.no_baseline else BASELINE_CSV
    subtitle = (f"ROC of each model — Youden and F1-max operating points with their score "
                f"thresholds ({args.flag} configuration for every model)" if args.flag else
                "ROC of each model on its best-AUC configuration — Youden and F1-max operating "
                "points with their score thresholds")
    plot_curves_points(records, os.path.join(args.out_dir, args.out_name), baseline, subtitle)
    save_summary(records, os.path.join(args.out_dir, args.summary_name))


if __name__ == '__main__':
    main()
