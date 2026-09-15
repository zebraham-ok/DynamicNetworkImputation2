#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed-ensemble summary: aggregate repeated runs (run_sampling.py --repeats N) into mean ± 95% CI.

Two complementary tools exist for seed ensembles:
  * Analysis/visualize_results.py --seed-ci / --multi-run confidence  -> TensorBoard event based
    (per-epoch curves with confidence bands, best-epoch metrics of the *restructured* results).
  * this script -> summary.yaml based: reads the per-seed summary.yaml that run_sampling.py writes
    next to each repeat, so the numbers are exactly the ones the trainer selected (no re-derivation
    from event files), and works without tensorboard.

Directory forms understood:
    results/<model>/sampling_<MMDD-HHMM>/summary.yaml            single run  (repeats = 1)
    results/<model>/sampling_<MMDD-HHMM>/seed<SEED>/summary.yaml one repeat of a seed ensemble

Donor rotation: with --repeats N, run_sampling.py shifts the mode order by one every repeat, so the
mode that supplies the backbone (the 'scratch' one) changes from repeat to repeat. The full mode key
('0_ftt_scratch' vs '0_ftt_frozen') is then no longer comparable across seeds, and the rows are
grouped by the mode itself ('0_ftt') instead, exactly like the 'by_mode' block of the run-level
summary.yaml; the role of every single run stays visible in the 'role' column of seed_metrics.csv.
Without rotation the full mode key is kept, because scratch and frozen runs are different regimes.

Outputs (default Analysis/npy_roc_output_ci/):
    seed_metrics.csv   long format: model, run, seed, role, mode, rotate_donor, metric, value
    seed_summary.csv   mean / std / 95% CI half-width / n per (model, run, mode, metric)
    seed_summary.md    human-readable table, one block per model and mode
    seed_ci_bars.png   grouped bar chart with error bars (Test AUC / Test F1 by default)

Usage:
    python Analysis/seed_ci_summary.py
    python Analysis/seed_ci_summary.py --results-dir results --out-dir Analysis/npy_roc_output_ci
    python Analysis/seed_ci_summary.py --metrics test_auc test_f1 test_loss --fig-metrics test_auc test_f1
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
import plot_models  # noqa: E402  (same directory; stdlib-only)

# Same two-sided 95% t quantiles as visualize_results.py::_T95_TABLE (small-sample seed ensembles)
_T95_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
              8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
              15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
              22: 2.074, 24: 2.064, 26: 2.056, 28: 2.048, 30: 2.042}

METRIC_LABELS = {
    'best_epoch': 'Best epoch',
    'test_auc': 'Test AUC',
    'test_f1': 'Test F1',
    'test_loss': 'Test loss',
    'train_bce': 'Train BCE',
    'factset_quantile_test': 'FSQ (test)',
    'wasserstein_diff_test': 'W1 (test)',
    'factset_quantile': 'FSQ (select)',
    'wasserstein_diff': 'W1 (select)',
    'epochs_run': 'Epochs run',
    'avg_epoch_seconds': 'Avg epoch (s)',
    'lr': 'LR at best epoch',
}

DEFAULT_METRICS = ['best_epoch', 'epochs_run', 'test_auc', 'test_f1', 'test_loss', 'train_bce',
                   'factset_quantile_test', 'wasserstein_diff_test']


def _t95(n):
    """Two-sided 95% t quantile for n samples (n-1 degrees of freedom)."""
    if n <= 1:
        return float('nan')
    df = n - 1
    if df >= 31:
        return 1.96
    if df in _T95_TABLE:
        return _T95_TABLE[df]
    smaller = [k for k in _T95_TABLE if k <= df]
    return _T95_TABLE[max(smaller)]


def mode_display(mode_key):
    """'1_ftf_frozen' -> 'ftf (frozen)'; '1_ftf' -> 'ftf' (rotated runs drop the role)."""
    parts = str(mode_key).split('_')
    if len(parts) >= 3 and parts[-1] in ('scratch', 'frozen'):
        return f"{'_'.join(parts[1:-1])} ({parts[-1]})"
    if len(parts) >= 2 and parts[0].isdigit():
        return '_'.join(parts[1:])
    return str(mode_key)


def _mode_label(mode_key):
    """'2_fff_frozen' -> '2_fff'.

    Mirror of run_sampling.mode_label(), which cannot be imported here because run_sampling pulls in
    torch while this script only needs numpy/pandas/yaml.
    """
    for suffix in ('_scratch', '_frozen'):
        if str(mode_key).endswith(suffix):
            return str(mode_key)[:-len(suffix)]
    return str(mode_key)


def _role_of(mode_key):
    """'scratch' (donor, trained from scratch) or 'frozen' (trained on another mode's backbone)."""
    return 'scratch' if str(mode_key).endswith('_scratch') else 'frozen'


def detect_rotated_runs(entries):
    """{(model, run): bool} -- does this run root rotate the backbone donor between its seeds?

    Taken from _meta.rotate_donor when present, and otherwise inferred from the summaries (more
    than one distinct scratch mode among the seeds), so hand-made directories are handled too.
    """
    hints, scratch_keys = {}, {}
    for entry in entries:
        key = (entry['model'], entry['run'])
        meta = entry['payload'].get('_meta', {}) or {}
        hints[key] = bool(meta.get('rotate_donor'))
        results = entry['payload'].get('results') or {}
        scratch_keys.setdefault(key, set()).update(k for k in results if str(k).endswith('_scratch'))
    return {k: bool(hints[k] or len(scratch_keys.get(k, ())) > 1) for k in hints}


def find_seed_summaries(results_root):
    """Locate every per-seed summary.yaml below results_root.

    Returns a list of dicts: {model, run, seed, path, payload}. Both the single-run layout
    (summary.yaml directly in the run directory) and the seed-ensemble layout (seed<SEED>/summary.yaml)
    are supported; a group-level summary.yaml that only carries an 'aggregate' block is skipped
    because its per-seed details live in the seed directories.
    """
    found = []
    root = Path(results_root)
    if not root.is_dir():
        return found

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        run_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir())
        for run_dir in run_dirs:
            seed_files = sorted(run_dir.glob('seed*/summary.yaml'))
            if seed_files:
                for path in seed_files:
                    seed_name = path.parent.name[4:]
                    found.append({'model': model_dir.name, 'run': run_dir.name,
                                  'seed': int(seed_name) if seed_name.isdigit() else None,
                                  'path': path})
            elif (run_dir / 'summary.yaml').exists():
                found.append({'model': model_dir.name, 'run': run_dir.name, 'seed': None,
                              'path': run_dir / 'summary.yaml'})

    for entry in found:
        with open(entry['path'], 'r', encoding='utf-8') as f:
            payload = yaml.safe_load(f) or {}
        entry['payload'] = payload
        if entry['seed'] is None:
            entry['seed'] = (payload.get('_meta', {}) or {}).get('seed')
    return found


def collect_rows(entries, metrics):
    """Flatten the per-seed summaries into long-format rows (skipping the aggregate-only files).

    The 'mode' column holds the full mode key ('1_ftf_frozen') unless the run rotates its donor, in
    which case it holds the mode itself ('1_ftf') so that the repeats stay comparable; the role of
    each individual run is always kept in the 'role' column.
    """
    rotated = detect_rotated_runs(entries)
    rows = []
    for entry in entries:
        results = entry['payload'].get('results') or {}
        if not results:
            continue
        meta = entry['payload'].get('_meta', {}) or {}
        run_key = (entry['model'], entry['run'])
        for mode_key, res in results.items():
            mode = _mode_label(mode_key) if rotated.get(run_key) else mode_key
            for metric in metrics:
                value = res.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                rows.append({
                    'model': entry['model'],
                    'run': entry['run'],
                    'seed': entry['seed'],
                    'repeats': meta.get('repeats'),
                    'selection_criterion': meta.get('selection_criterion') or None,
                    'rotate_donor': bool(rotated.get(run_key)),
                    'donor_mode': meta.get('donor_mode') or None,
                    'role': _role_of(mode_key),
                    'mode': mode,
                    'metric': metric,
                    'value': float(value),
                })
    return pd.DataFrame(rows)


def summarise(df):
    """mean / std / 95% CI half-width / n per (model, run, mode, metric)."""
    rows = []
    for key, group in df.groupby(['model', 'run', 'mode', 'metric'], sort=False):
        values = group['value'].dropna().to_numpy(dtype=float)
        n = len(values)
        if n == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else np.nan
        ci95 = float(_t95(n) * std / np.sqrt(n)) if (n > 1 and np.isfinite(std)) else np.nan
        rows.append({
            'model': key[0], 'run': key[1], 'mode': key[2], 'metric': key[3],
            'mean': mean, 'std': std, 'ci95': ci95, 'n': n,
            'min': float(values.min()), 'max': float(values.max()),
            'rotate_donor': bool(group['rotate_donor'].iloc[0]) if 'rotate_donor' in group else False,
            'roles': ','.join(sorted(set(group['role'].dropna()))) if 'role' in group else '',
            'seeds': ', '.join(str(int(s)) for s in sorted(set(group['seed'].dropna()))),
        })
    return pd.DataFrame(rows)


def _fmt(mean, ci95, n, metric):
    """'0.9012 ± 0.0031' (or a bare value when a single run leaves the CI undefined)."""
    decimals = 0 if metric in ('best_epoch', 'epochs_run') else 4
    if metric == 'avg_epoch_seconds':
        decimals = 1
    if not np.isfinite(mean):
        return '-'
    if n is None or n < 2 or not np.isfinite(ci95):
        return f"{mean:.{decimals}f} (n=1)"
    return f"{mean:.{decimals}f} ± {ci95:.{decimals}f}"


def write_markdown(summary_df, path, metrics):
    """One block per (model, run, mode): metric | mean ± 95% CI | range | n | seeds."""
    lines = ['# Seed-ensemble summary (mean ± 95% CI across repeated runs)', '']
    if summary_df.empty:
        lines.append('_No per-seed summary.yaml found._')
    for (model, run, mode), group in summary_df.groupby(['model', 'run', 'mode'], sort=False):
        subset = group.set_index('metric')
        n_max = int(subset['n'].max())
        seeds = sorted({s for s in subset['seeds'] if s})
        lines.append(f"## {model} — {mode_display(mode)}")
        lines.append('')
        lines.append(f"- run directory: `{model}/{run}`")
        lines.append(f"- repeats: {n_max}" + (f" (seeds {', '.join(seeds)})" if seeds else ''))
        if 'rotate_donor' in subset.columns and bool(subset['rotate_donor'].iloc[0]):
            roles = subset['roles'].iloc[0] if 'roles' in subset.columns else 'scratch,frozen'
            lines.append(f"- **donor rotation**: this run shifts the backbone donor between the "
                         f"repeats, so the mean pools the runs of this mode with roles "
                         f"`{roles}` (every mode donates the backbone once); the role of each "
                         f"single run is the `role` column of seed_metrics.csv")
        lines.append('')
        lines.append('| Metric | Mean ± 95% CI | Min – Max | n |')
        lines.append('|---|---|---|---|')
        for metric in metrics:
            if metric not in subset.index:
                continue
            row = subset.loc[metric]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rng = (f"{row['min']:.4f} – {row['max']:.4f}"
                   if np.isfinite(row['min']) and np.isfinite(row['max']) else '-')
            lines.append(f"| {METRIC_LABELS.get(metric, metric)} "
                         f"| {_fmt(row['mean'], row['ci95'], row['n'], metric)} "
                         f"| {rng} | {int(row['n'])} |")
        lines.append('')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def plot_bars(summary_df, path, fig_metrics, labels=None):
    """Grouped bar chart: one bar per (model, mode), error bar = 95% CI across seeds.

    `labels` is the ordered model -> display-name mapping of the plot config; its key order drives
    the bar/colour order and its values the legend text. Without it the directory names are used.
    """
    if summary_df.empty:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_metrics = [m for m in fig_metrics if m in summary_df['metric'].unique()]
    if not fig_metrics:
        print("  [WARN] none of the requested figure metrics are present; skipping the bar chart")
        return

    labels = labels or {}
    modes = list(dict.fromkeys(summary_df['mode']))
    present = set(summary_df['model'])
    models = [m for m in labels if m in present] if labels else list(dict.fromkeys(summary_df['model']))
    models += [m for m in dict.fromkeys(summary_df['model']) if m not in models]
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, len(fig_metrics), figsize=(4.6 * len(fig_metrics), 4.4), squeeze=False)

    for ax_idx, metric in enumerate(fig_metrics):
        ax = axes[0][ax_idx]
        width = 0.8 / max(len(models), 1)
        for m_idx, model in enumerate(models):
            xs, ys, errs = [], [], []
            for x_idx, mode in enumerate(modes):
                row = summary_df[(summary_df['model'] == model) &
                                 (summary_df['mode'] == mode) &
                                 (summary_df['metric'] == metric)]
                if row.empty:
                    continue
                row = row.iloc[0]
                xs.append(x_idx + m_idx * width)
                ys.append(row['mean'])
                errs.append(row['ci95'] if np.isfinite(row['ci95']) else 0.0)
            if xs:
                ax.bar(xs, ys, width=width, yerr=errs, capsize=3,
                       color=cmap(m_idx % 10), alpha=0.85, label=labels.get(model, model),
                       error_kw={'linewidth': 1.0, 'ecolor': 'dimgray'})
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(modes))])
        ax.set_xticklabels([mode_display(m) for m in modes], fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} (mean ± 95% CI)", fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle('Seed ensemble: repeated runs with different initialisations', fontsize=13,
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Bar chart                     : {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate seed-ensemble runs into mean ± 95% CI')
    parser.add_argument('--results-dir', default=os.path.join(_REPO_ROOT, 'results'),
                        help='directory holding <model>/sampling_*/[seed*]/summary.yaml '
                             '(default: opensource-revise/results)')
    parser.add_argument('--out-dir', default=os.path.join(_SCRIPT_DIR, 'npy_roc_output_ci'),
                        help='output directory (default: Analysis/npy_roc_output_ci)')
    parser.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                        help='metrics to tabulate (must exist in summary.yaml)')
    parser.add_argument('--fig-metrics', nargs='+', default=['test_auc', 'test_f1'],
                        help='metrics to draw in the bar chart')
    parser.add_argument('--no-figure', action='store_true', help='skip the bar chart')
    parser.add_argument('--model-config', default=None,
                        help='plot list deciding which models are aggregated / how they are named '
                             f'(default: {plot_models.CONFIG_NAME} next to the results root)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Seed-ensemble summary")
    print(f"  results : {args.results_dir}")
    print(f"  output  : {args.out_dir}")

    entries = find_seed_summaries(args.results_dir)
    if not entries:
        print("  [ERROR] no summary.yaml found under results/<model>/sampling_*/ — "
              "run SampleSetting/run_sampling.py first")
        return

    # Same plot list as the other figures: comment a line there and the model leaves the CSVs, the
    # markdown table, the console digest and the bar chart. Its order drives the bar colours.
    cfg = plot_models.load_plot_models(args.results_dir, args.model_config)
    labels = {}
    if cfg is not None:
        labels = cfg.mapping(available={e['model'] for e in entries})
        entries = [e for e in entries if e['model'] in labels]
        if not entries:
            print("  [ERROR] the plot config selects no model present in this tree")
            return

    runs = {(e['model'], e['run']) for e in entries}
    seeds_per_run = {}
    for e in entries:
        seeds_per_run.setdefault((e['model'], e['run']), set()).add(e['seed'])
    multi = {k: v for k, v in seeds_per_run.items() if len(v) > 1}
    print(f"  found   : {len(entries)} summaries in {len(runs)} run roots; "
          f"{len(multi)} of them hold more than one seed")
    if not multi:
        print("  [INFO] every run root holds a single seed — the CIs will read 'n=1'. "
              "Use run_sampling.py --repeats N to produce a seed ensemble.")
    rotating = sorted(f"{m}/{r}" for (m, r), v in detect_rotated_runs(entries).items() if v)
    if rotating:
        print(f"  [INFO] donor rotation in {len(rotating)} run root(s) "
              f"({', '.join(rotating)}): the mode that supplies the backbone changes between the "
              f"repeats, so rows are grouped by mode (scratch/frozen pooled) and the role of every "
              f"single run is kept in the 'role' column of seed_metrics.csv")

    rows_df = collect_rows(entries, args.metrics)
    if rows_df.empty:
        print("  [ERROR] the summaries carry none of the requested metrics: "
              f"{', '.join(args.metrics)}")
        return
    summary_df = summarise(rows_df)

    metrics_path = os.path.join(args.out_dir, 'seed_metrics.csv')
    summary_path = os.path.join(args.out_dir, 'seed_summary.csv')
    md_path = os.path.join(args.out_dir, 'seed_summary.md')
    rows_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    write_markdown(summary_df, md_path, args.metrics)
    print(f"  Per-seed values               : {metrics_path}")
    print(f"  Aggregate (mean/std/ci95/n)   : {summary_path}")
    print(f"  Markdown table                : {md_path}")

    if not args.no_figure:
        plot_bars(summary_df, os.path.join(args.out_dir, 'seed_ci_bars.png'), args.fig_metrics,
                  labels=labels)

    # Console digest: one line per (model, run root, mode) for the headline metrics
    print(f"\n  {'Model':<10} {'Mode':<16} {'Test AUC':<20} {'Test F1':<20} "
          f"{'FSQ (test)':<20} {'n':<3}")
    print(f"  {'-' * 95}")
    for (model, run, mode), group in summary_df.groupby(['model', 'run', 'mode'], sort=False):
        subset = group.set_index('metric')

        def cell(metric):
            if metric not in subset.index:
                return '-'
            row = subset.loc[metric]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return _fmt(row['mean'], row['ci95'], row['n'], metric)

        n_max = int(subset['n'].max())
        print(f"  {model:<10} {mode_display(mode):<16} {cell('test_auc'):<20} "
              f"{cell('test_f1'):<20} {cell('factset_quantile_test'):<20} {n_max:<3}")
    print(f"\n  Run roots: {', '.join(f'{m}/{r}' for m, r in sorted(runs))}")


if __name__ == '__main__':
    main()
