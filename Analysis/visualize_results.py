#!/usr/bin/env python3
"""
Results visualization script: reads TensorBoard event files under results/ and produces line plots and heatmaps.
Usage: python Analysis/visualize_results.py
Output: Analysis/visualization/
"""

import os
import sys
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import seaborn as sns
from collections import defaultdict
from pathlib import Path

# Set CJK font
for _font_name in ['Microsoft YaHei', 'SimHei', 'DengXian']:
    try:
        fm.findfont(_font_name, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_font_name] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE_DIR = Path(__file__).resolve().parent.parent  # repository root
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = BASE_DIR / 'Analysis' / 'visualization'

# SampleSetting/sample_setting.yaml is the single source of truth for the ftt/ftf/fff/tff
# switch combinations; the legend below is rendered from it so the two can never drift apart.
SAMPLE_SETTING_YAML = BASE_DIR / 'SampleSetting' / 'sample_setting.yaml'
FLAG_ORDER = ['ftt', 'ftf', 'fff', 'tff']
# Task-level annotation, not a config switch: ftt > ftf > fff > tff in negative-sampling difficulty.
FLAG_HINT = {'ftt': ' (hardest)', 'tff': ' (easiest)'}

# Negative-sampling flag -> long label
FLAG_LABELS = {
    'ftt': 'Filter x  Industry v  NoSup v (hardest)',
    'ftf': 'Filter x  Industry v  NoSup x',
    'fff': 'Filter x  Industry x  NoSup x',
    'tff': 'Filter v  Industry x  NoSup x (easiest)',
}

# Negative-sampling flag -> short label
FLAG_SHORT = {
    'ftt': 'ftt',
    'ftf': 'ftf',
    'fff': 'fff',
    'tff': 'tff',
}

DIFFICULTY_ORDER = {'ftt': 0, 'ftf': 1, 'fff': 2, 'tff': 3}

MODEL_DISPLAY = {
    'bigru': 'BiGRU',
    'egcn': 'EvolveGCN-H',
    'gatgru': 'GAT-GRU',
    'tna': 'BiTNA',
    'seal': 'SEAL',
}

MODEL_BASE_COLORS = {
    'bigru': 'Blues',
    'egcn': 'Greens',
    'gatgru': 'Oranges',
    'tna': 'Purples',
    'seal': 'RdPu',
}

# Shade position of each of the 4 configs per model (deep->light, i.e. ftt->tff)
SHADE_POSITIONS = [0.9, 0.7, 0.575, 0.45]

# Per-model spec: {concept: actual tag}
# best_criterion is the validation AUC: under the standard protocol the validation split selects
# the epoch and the test split is only reported. Runs produced before that switch have no
# Val/Epoch_AUC curve, in which case get_best_epoch_metrics() falls back to Test/Epoch_AUC.
MODEL_METRIC_TAGS = {
    'bigru': {
        'train_loss': 'Train/Epoch_BCE',
        'train_loss2': 'Train/Epoch_Margin',
        'test_auc': 'Test/Epoch_AUC',
        'test_f1': 'Test/Epoch_F1',
        'test_loss': 'Test/Epoch_Loss',
        'best_criterion': 'Val/Epoch_AUC',  # fallback: Test/Epoch_AUC (legacy runs)
        'best_direction': 'max',
    },
    'egcn': {
        'train_loss': 'Train/Epoch_BCE',
        'train_loss2': 'Train/Epoch_Margin',
        'test_auc': 'Test/Epoch_AUC',
        'test_f1': 'Test/Epoch_F1',
        'test_loss': 'Test/Epoch_Loss',
        'best_criterion': 'Val/Epoch_AUC',
        'best_direction': 'max',
    },
    'gatgru': {
        'train_loss': 'Train/Epoch_BCE',
        'train_loss2': 'Train/Epoch_Margin',
        'test_auc': 'Test/Epoch_AUC',
        'test_f1': 'Test/Epoch_F1',
        'test_loss': 'Test/Epoch_Loss',
        'best_criterion': 'Val/Epoch_AUC',
        'best_direction': 'max',
    },
    'tna': {
        'train_loss': 'Train/Epoch_BCE',
        'train_loss2': 'Train/Epoch_Margin',
        'test_auc': 'Test/Epoch_AUC',
        'test_f1': 'Test/Epoch_F1',
        'test_loss': 'Test/Epoch_Loss',
        'best_criterion': 'Val/Epoch_AUC',
        'best_direction': 'max',
    },
    # seal: step-level records; tag names aligned with other models (epoch column actually stores batch step count)
    'seal': {
        'train_loss': 'Train/Epoch_BCE',
        'train_loss2': None,
        'test_auc': 'Test/Epoch_AUC',
        'test_f1': 'Test/Epoch_F1',
        'test_loss': 'Test/Epoch_Loss',
        'best_criterion': 'Val/Epoch_AUC',  # fallback: Test/Epoch_AUC (runs without a val split)
        'best_direction': 'max',
    },
}

METRIC_DISPLAY = {
    'train_loss': 'Train Loss',
    'test_auc': 'AUC',
    'test_f1': 'F1',
    'test_loss': 'Test/Val Loss',
}

# When the same model+flag has multiple runs: 'longest' keeps only the run with the most epochs,
# 'confidence' keeps all runs (every agg then draws mean ± std as a confidence band).
# Seed ensembles (run_sampling.py --repeats N) land in exactly this case: N runs per (model, flag),
# one per seed, so 'confidence' is what turns the seed ensemble into CIs on the curves.
# Override per invocation with --multi-run, or with the IMPUT_MULTI_RUN environment variable.
MULTI_RUN_MODE = os.environ.get('IMPUT_MULTI_RUN', 'longest')
if MULTI_RUN_MODE not in ('longest', 'confidence'):
    print(f"  [WARN] unknown IMPUT_MULTI_RUN={MULTI_RUN_MODE!r}, falling back to 'longest'")
    MULTI_RUN_MODE = 'longest'


def _filter_multi_run(df, mode='longest'):
    """
    Handle the case where a single (model, flag) pair has multiple run records.
    mode='longest':   keep only the one run_id with the most unique epochs per group
    mode='confidence': keep all runs, no filtering (each curve becomes mean ± std over runs)
    """
    if mode == 'confidence':
        kept = df.copy()
        # Make the aggregation visible: a band is only meaningful if it really pools several runs
        counts = kept.groupby(['model', 'flag'])['run_id'].nunique()
        multi = counts[counts > 1]
        if len(multi):
            print(f"  Multi-run aggregation (confidence): {len(multi)} (model, flag) groups pool "
                  f"several runs -> curves are drawn as mean ± std")
            if 'seed' in kept.columns and kept['seed'].notna().any():
                seeds = sorted({int(s) for s in kept['seed'].dropna().unique()})
                print(f"    seeds pooled: {seeds}")
        else:
            print("  Multi-run aggregation (confidence): no (model, flag) has more than one run, "
                  "bands will be empty")
        return kept

    run_epoch_counts = (
        df.groupby(['model', 'flag', 'run_id'])['epoch']
        .nunique()
        .reset_index(name='n_epochs')
    )
    best_runs = run_epoch_counts.loc[
        run_epoch_counts.groupby(['model', 'flag'])['n_epochs'].idxmax(),
        ['model', 'flag', 'run_id']
    ]

    filtered = df.merge(best_runs, on=['model', 'flag', 'run_id'], how='inner')

    n_orig = df['run_id'].nunique()
    n_kept = filtered['run_id'].nunique()
    if n_orig > n_kept:
        dropped = n_orig - n_kept
        print(f"  Multi-run filter ({mode}): kept {n_kept}/{n_orig} runs (dropped {dropped} shorter runs)")

    return filtered


# A restructured run directory is "<MMDD-HHMM>-<flag>", optionally carrying the seed of a repeated
# run: "0912-1118-fff", "0912-1118-fff-s3" or "0912-1118-fff-seed3". Two seeds of the same
# model+flag therefore appear as two runs, which is exactly what MULTI_RUN_MODE='confidence'
# aggregates into a confidence band (see seed_ci_summary.py for the run-level directory form).
RUN_NAME_RE = re.compile(r'(\d{4}-\d{4})-(f\w{2}|t\w{2})(?:-s(?:eed)?(\d+))?$')


def parse_run_name(dirname):
    """
    Parse a directory name, e.g. "0714-1642-ftt" -> ('0714-1642', 'ftt')
    An optional seed suffix ("0714-1642-ftt-s3") is tolerated and ignored here; use
    parse_run_seed() to read it. The timestamp stays the run_id, so two seeds never collide.
    """
    match = RUN_NAME_RE.match(dirname)
    if match:
        return match.group(1), match.group(2)
    return None, None


def parse_run_seed(dirname):
    """Seed encoded in a run directory name ("...-ftt-s3" / "...-ftt-seed3"), else None."""
    match = RUN_NAME_RE.match(dirname)
    if match and match.group(3) is not None:
        return int(match.group(3))
    return None


def extract_scalars(event_dir, include_wall_time=False):
    """
    Extract all scalar data from the TF Event file.
    Returns {tag: [(step, value, wall_time?), ...], ...}
    If include_wall_time=True, each element is (step, value, wall_time).
    """
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    result = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            if include_wall_time:
                result[tag] = [(e.step, e.value, e.wall_time) for e in events]
            else:
                result[tag] = [(e.step, e.value) for e in events]
        except Exception:
            pass
    return result


def read_recorded_epochs(run_dir):
    """Number of epochs a run actually trained, as recorded by the run itself.

    SEAL writes `run_summary.json` (`epochs`); `run_sampling.py` writes `summary.yaml`
    (`results.<mode>.epochs_run` per mode, `_meta.epochs_run` for the aggregate).
    Returns None when neither file is present / readable (e.g. a restructured run directory).
    """
    import json
    run_dir = Path(run_dir)

    json_path = run_dir / 'run_summary.json'
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
            value = data.get('epochs')
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        except Exception:
            pass

    yaml_path = run_dir / 'summary.yaml'
    if yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
            candidates = []
            for entry in (data.get('results') or {}).values():
                if isinstance(entry, dict) and entry.get('epochs_run') is not None:
                    candidates.append(entry['epochs_run'])
            if (data.get('_meta') or {}).get('epochs_run') is not None:
                candidates.append(data['_meta']['epochs_run'])
            candidates = [int(c) for c in candidates if isinstance(c, (int, float)) and c > 0]
            if candidates:
                return max(candidates)
        except Exception:
            pass

    return None


def compute_epoch_duration_from_walltime(scalars_wt, model='', num_epochs=None):
    """
    Estimate per-epoch duration from event wall_time (wall_time difference between
    consecutive epoch summary events); returns the mean epoch duration (seconds).
    Uses the timestamps of epoch-level metrics such as Train/Epoch_BCE or Test/Epoch_AUC;
    returns a valid value only when consecutive epochs >= 2 and the mean difference is reasonable.

    Special handling for the SEAL model: its Epoch_* metrics are recorded per
    periodic_eval_steps (default 1000) rather than per epoch and must be rescaled
    by the total number of steps. `num_epochs` is the epoch count recorded by the run
    (see read_recorded_epochs, run_summary.json / summary.yaml); without it no rescaling is
    possible and the raw inter-event gap is returned, because the batch-to-epoch ratio cannot be
    inferred from the scalars (SEAL logs nothing at epoch granularity except Time/Epoch_Duration,
    and this fallback only runs when that tag is missing).
    """
    candidate_tags = ['Train/Epoch_BCE', 'Test/Epoch_AUC', 'Test/Epoch_F1',
                      'Train/Epoch_F1', 'Test/Epoch_Loss']
    durations = []
    selected_tag = None
    for tag in candidate_tags:
        if tag not in scalars_wt:
            continue
        events = scalars_wt[tag]
        events_sorted = sorted(events, key=lambda x: x[0])
        if len(events_sorted) < 2:
            continue
        tag_durations = []
        for i in range(1, len(events_sorted)):
            dt = events_sorted[i][2] - events_sorted[i - 1][2]
            if 0 < dt < 86400:  # filter outliers (>24h usually means an interrupt/restart)
                tag_durations.append(dt)
        if len(tag_durations) >= 2:
            durations.extend(tag_durations)
            selected_tag = events_sorted
            break

    if not durations or selected_tag is None:
        return None

    avg_dt = float(np.mean(durations))

    # SEAL special handling: Train/Epoch_* is recorded per batch step and must be rescaled to epoch
    # duration. The epoch count is NOT hard-coded any more: SEAL trains until early stopping (the
    # epoch ceiling comes from Training/common_config.yaml, not from Models/configs/seal.yaml), so
    # it is read from the run's own record.
    if model == 'seal':
        if not num_epochs or num_epochs < 1:
            return avg_dt
        max_step = selected_tag[-1][0]
        step_gap = selected_tag[1][0] - selected_tag[0][0]
        if step_gap <= 0:
            return avg_dt

        batches_per_epoch = max_step / num_epochs

        epoch_time = avg_dt * (batches_per_epoch / step_gap)

        if 0 < epoch_time < 86400:
            return epoch_time
        return avg_dt

    return avg_dt


def collect_all_data():
    """
    Walk all models/runs under results/ and extract data.
    Returns a DataFrame: [model, run_id, flag, epoch, tag, value]
    Also estimates and injects Time/Epoch_Duration from wall_time.
    """
    records = []
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if model not in MODEL_DISPLAY:
            continue

        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            timestamp, flag = parse_run_name(run_dir.name)
            if flag is None:
                continue

            scalars_wt = extract_scalars(run_dir, include_wall_time=True)
            seed = parse_run_seed(run_dir.name)
            for tag, events in scalars_wt.items():
                for step, value, _wall_time in events:
                    records.append({
                        'model': model,
                        'run_id': run_dir.name,
                        'flag': flag,
                        'seed': seed,
                        'epoch': step,
                        'tag': tag,
                        'value': value,
                    })

            # Estimate from wall_time and inject only when TensorBoard itself did not record this tag
            if 'Time/Epoch_Duration' not in scalars_wt:
                # SEAL needs the number of epochs the run actually trained (it is no longer fixed),
                # which only the run's own summary knows.
                avg_dur = compute_epoch_duration_from_walltime(
                    scalars_wt, model=model, num_epochs=read_recorded_epochs(run_dir))
            else:
                avg_dur = None
            if avg_dur is not None:
                records.append({
                    'model': model,
                    'run_id': run_dir.name,
                    'flag': flag,
                    'seed': seed,
                    'epoch': 0,  # aggregate value, step not applicable
                    'tag': 'Time/Epoch_Duration',
                    'value': avg_dur,
                })

    df = pd.DataFrame(records)

    df = df[df['model'].isin(MODEL_DISPLAY)]

    df = _filter_seal_mixed_wasserstein(df)

    return df


def _filter_seal_mixed_wasserstein(df):
    """
    SEAL records Monitor/Wasserstein_* scalars both per epoch (0,1,2...) and per
    batch step (100,200...). If both exist for a (run_id, tag), drop the low-step records.
    """
    if df.empty:
        return df

    mask = (
        (df['model'] == 'seal') &
        df['tag'].str.startswith('Monitor/Wasserstein_')
    )
    seal_df = df[mask].copy()
    if seal_df.empty:
        return df

    keep_idx = []
    for (run_id, tag), group in seal_df.groupby(['run_id', 'tag']):
        steps = group['epoch'].values
        has_epoch = (steps < 100).any()
        has_batch = (steps >= 100).any()
        if has_epoch and has_batch:
            keep = group[group['epoch'] >= 100]
        else:
            keep = group
        keep_idx.extend(keep.index.tolist())

    seal_kept = df.loc[keep_idx]
    other_df = df[~mask]
    return pd.concat([other_df, seal_kept], ignore_index=True)


def get_best_epoch_metrics(df):
    """
    For each (model, flag) pair, select the best epoch by the model's best criterion,
    and return all key metrics at that epoch.
    Each model's best criterion is defined in MODEL_METRIC_TAGS.
    """
    best_rows = []
    for model in df['model'].unique():
        mt = MODEL_METRIC_TAGS.get(model, {})
        criterion_tag = mt.get('best_criterion', 'Test/Epoch_AUC')
        direction = mt.get('best_direction', 'max')

        criterion_df = df[(df['tag'] == criterion_tag) & (df['model'] == model)]
        if criterion_df.empty:
            for fb_tag, fb_dir in [('Test/Epoch_AUC', 'max'), ('Test/Epoch_F1', 'max'),
                                    ('Val/Epoch_Precision', 'max'), ('Val/Epoch_Loss', 'min'),
                                    ('Train/Epoch_Loss', 'min')]:
                criterion_df = df[(df['tag'] == fb_tag) & (df['model'] == model)]
                if not criterion_df.empty:
                    criterion_tag, direction = fb_tag, fb_dir
                    break
            if criterion_df.empty:
                continue

        for flag, group in criterion_df.groupby('flag'):
            if direction == 'max':
                best = group.loc[group['value'].idxmax()]
            else:
                best = group.loc[group['value'].idxmin()]
            best_epoch = best['epoch']

            epoch_data = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['epoch'] == best_epoch)]
            for _, row in epoch_data.iterrows():
                best_rows.append({
                    'model': model,
                    'flag': flag,
                    'best_epoch': best_epoch,
                    'tag': row['tag'],
                    'value': row['value'],
                })

    return pd.DataFrame(best_rows)


def get_best_epoch_metrics_per_run(df):
    """Per (model, flag, run_id) best-epoch metrics, i.e. get_best_epoch_metrics split per run.

    This is the honest way to summarise a seed ensemble: each run selects its own best epoch using
    its own criterion, and only the selected values are averaged afterwards. Taking a single
    argmax over all runs of a (model, flag) would instead report the best of N seeds, which is an
    upward-biased (order-statistic) number rather than an estimate of typical performance.
    """
    best_rows = []
    for model in df['model'].unique():
        mt = MODEL_METRIC_TAGS.get(model, {})
        criterion_tag = mt.get('best_criterion', 'Test/Epoch_AUC')
        direction = mt.get('best_direction', 'max')

        criterion_df = df[(df['tag'] == criterion_tag) & (df['model'] == model)]
        if criterion_df.empty:
            for fb_tag, fb_dir in [('Test/Epoch_AUC', 'max'), ('Test/Epoch_F1', 'max'),
                                   ('Val/Epoch_Precision', 'max'), ('Val/Epoch_Loss', 'min'),
                                   ('Train/Epoch_Loss', 'min')]:
                criterion_df = df[(df['tag'] == fb_tag) & (df['model'] == model)]
                if not criterion_df.empty:
                    criterion_tag, direction = fb_tag, fb_dir
                    break
            if criterion_df.empty:
                continue

        for (flag, run_id), group in criterion_df.groupby(['flag', 'run_id']):
            if direction == 'max':
                best = group.loc[group['value'].idxmax()]
            else:
                best = group.loc[group['value'].idxmin()]
            best_epoch = best['epoch']

            epoch_data = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['run_id'] == run_id) &
                            (df['epoch'] == best_epoch)]
            for _, row in epoch_data.iterrows():
                best_rows.append({
                    'model': model,
                    'flag': flag,
                    'run_id': run_id,
                    'seed': row.get('seed'),
                    'best_epoch': best_epoch,
                    'tag': row['tag'],
                    'value': row['value'],
                })

    return pd.DataFrame(best_rows)


# Metrics worth reporting as a seed aggregate (subset of the tags in the per-run table)
SEED_CI_TAGS = ['Test/Epoch_AUC', 'Test/Epoch_F1', 'Test/Epoch_Loss',
                'Monitor/Factset_Quantile', 'Monitor/Wasserstein_Diff']


def aggregate_best_metrics(per_run_df, ci=True):
    """Collapse the per-run best metrics into mean ± 95% CI per (model, flag, tag).

    Returns a frame with the same columns as get_best_epoch_metrics (so plot_summary_table can
    consume it directly, with 'value' = mean over runs) plus 'std', 'ci95', 'n' and 'best_epoch'
    (mean of the per-run best epochs).
    """
    if per_run_df is None or per_run_df.empty:
        return pd.DataFrame()

    rows = []
    for (model, flag, tag), group in per_run_df.groupby(['model', 'flag', 'tag']):
        values = group['value'].dropna().to_numpy(dtype=float)
        n = len(values)
        if n == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else np.nan
        if ci and n > 1 and np.isfinite(std):
            ci95 = float(_T95_TABLE.get(n - 1, 1.96) * std / np.sqrt(n))
        else:
            ci95 = np.nan
        rows.append({
            'model': model,
            'flag': flag,
            'tag': tag,
            'value': mean,
            'std': std,
            'ci95': ci95,
            'n': n,
            'best_epoch': float(group['best_epoch'].mean()),
        })
    return pd.DataFrame(rows)


def report_seed_ci(per_run_df, err_df, out_dir=None):
    """Print and persist the across-run aggregate that the paper table is built from.

    Writes <out_dir>/seed_ci_per_run.csv (one row per run: the value selected in its own best
    epoch) and <out_dir>/seed_ci_summary.csv (mean, std, 95% CI half-width, number of runs), so the
    numbers in the manuscript can always be traced back to the individual repeats.
    """
    out_dir = out_dir or OUTPUT_DIR
    if per_run_df is None or per_run_df.empty or err_df is None or err_df.empty:
        print("  [WARN] no per-run best-epoch metrics found; skipping the seed aggregate")
        return None

    runs_per_group = per_run_df.groupby(['model', 'flag'])['run_id'].nunique()
    seeds = []
    if 'seed' in per_run_df.columns and per_run_df['seed'].notna().any():
        seeds = sorted({int(s) for s in per_run_df['seed'].dropna().unique()})
    print(f"  Runs per (model, flag): min={int(runs_per_group.min())}, "
          f"max={int(runs_per_group.max())}"
          + (f" | seeds: {seeds}" if seeds else
             " | no seed suffix in the run names (add '-s<seed>' when restructuring)"))
    if int(runs_per_group.min()) < 2:
        print("  [WARN] at least one (model, flag) has a single run -> its CI is undefined ('n/a')")
    if seeds:
        mix = per_run_df.groupby(['model', 'flag'])['seed'].agg(
            lambda s: bool(s.notna().any() and s.isna().any()))
        mix = mix[mix]
        if len(mix):
            print("  [WARN] these (model, flag) groups mix seeded and unseeded runs, so the aggregate "
                  "pools a seed ensemble with a separate run: "
                  + ', '.join(f"{m}/{f}" for m, f in mix.index))

    print(f"\n  {'Method':<12} {'Flag':<6} {'Metric':<28} {'Mean':>10} {'±95%CI':>9} {'Runs':>5}")
    print(f"  {'-' * 76}")
    for _, r in err_df[err_df['tag'].isin(SEED_CI_TAGS)].sort_values(
            ['model', 'flag', 'tag']).iterrows():
        ci = 'n/a' if not np.isfinite(r['ci95']) else f"{r['ci95']:.4f}"
        print(f"  {MODEL_DISPLAY.get(r['model'], r['model']):<12} "
              f"{FLAG_SHORT.get(r['flag'], r['flag']):<6} {r['tag']:<28} "
              f"{r['value']:>10.4f} {ci:>9} {int(r['n']):>5}")

    os.makedirs(out_dir, exist_ok=True)
    per_run_path = os.path.join(out_dir, 'seed_ci_per_run.csv')
    summary_path = os.path.join(out_dir, 'seed_ci_summary.csv')
    per_run_df.to_csv(per_run_path, index=False)
    err_df.to_csv(summary_path, index=False)
    print(f"\n  Per-run best metrics            : {per_run_path}")
    print(f"  Seed aggregate (mean ± 95% CI)  : {summary_path}")
    return err_df


# Two-sided 95% t quantiles by degrees of freedom (small-sample table for seed ensembles)
_T95_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
              8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
              15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
              22: 2.074, 24: 2.064, 26: 2.056, 28: 2.048, 30: 2.042}


def get_model_colors():
    """
    Build a color map for each (model, flag).
    Same colormap per model; flags go from dark to light with difficulty.
    """
    colors = {}
    for model, cmap_name in MODEL_BASE_COLORS.items():
        cmap = plt.get_cmap(cmap_name)
        for flag, pos in zip(DIFFICULTY_ORDER.keys(), SHADE_POSITIONS):
            colors[(model, flag)] = cmap(pos)
    return colors


def plot_epoch_lines(df, tag_patterns, title, filename, ylabel, group_by='model',
                     ncols=2, figsize=None):
    """
    Generic epoch line plot (SEAL uses dual x-axes: Epoch at bottom, Step at top).
    tag_patterns: [(tag_regex, sub_title), ...]
    """
    colors = get_model_colors()
    n_plots = len(tag_patterns)
    if figsize is None:
        figsize = (7 * ncols, 5 * ((n_plots + ncols - 1) // ncols))

    seal_present = 'seal' in df['model'].unique()

    fig, axes = plt.subplots(
        (n_plots + ncols - 1) // ncols, ncols,
        figsize=figsize, squeeze=False
    )
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

    for idx, (tag_pattern, sub_title) in enumerate(tag_patterns):
        ax = axes[idx // ncols][idx % ncols]
        ax_epoch = ax
        ax_step = ax.twiny() if seal_present else None

        max_epoch = 0
        max_step = 0
        seal_lines = []
        epoch_lines = []

        for model in sorted(df['model'].unique()):
            for flag in DIFFICULTY_ORDER:
                subset = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['tag'].str.match(tag_pattern))]
                if subset.empty:
                    continue
                agg = subset.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
                agg = agg.sort_values('epoch')
                color = colors.get((model, flag), 'gray')
                label = f"{MODEL_DISPLAY.get(model, model)}-{FLAG_SHORT[flag]}"

                if model == 'seal':
                    max_step = max(max_step, agg['epoch'].max())
                    seal_lines.append((agg, color, label))
                else:
                    agg_ep = agg[agg['epoch'] <= 20]
                    if not agg_ep.empty:
                        max_epoch = max(max_epoch, agg_ep['epoch'].max())
                    epoch_lines.append((agg, color, label))

        if max_epoch > 0:
            ax_epoch.set_xlim(0, max_epoch)
        if max_step > 0 and ax_step:
            ax_step.set_xlim(0, max_step)

        # Non-SEAL models: epoch axis at bottom, solid lines
        for agg, color, label in epoch_lines:
            ax_epoch.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.5,
                          label=label, alpha=0.9)
            if len(agg) > 1 and agg['std'].notna().any():
                ax_epoch.fill_between(agg['epoch'],
                                      agg['mean'] - agg['std'],
                                      agg['mean'] + agg['std'],
                                      color=color, alpha=0.15)

        # SEAL: step axis at top, dashed lines
        if ax_step:
            for agg, color, label in seal_lines:
                ax_step.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.8,
                             label=label, alpha=0.9, linestyle='--')
                if len(agg) > 1 and agg['std'].notna().any():
                    ax_step.fill_between(agg['epoch'],
                                         agg['mean'] - agg['std'],
                                         agg['mean'] + agg['std'],
                                         color=color, alpha=0.08)

        ax_epoch.set_xlabel('Epoch', fontsize=11)
        if ax_step:
            ax_step.set_xlabel('Step (SEAL only)', fontsize=11)
        ax_epoch.set_ylabel(ylabel, fontsize=11)
        ax_epoch.tick_params(labelsize=9)
        if ax_step:
            ax_step.tick_params(labelsize=9)

        ax.set_title(sub_title, fontsize=14)
        ax.grid(True, alpha=0.3)

        handles1, labels1 = ax_epoch.get_legend_handles_labels()
        handles2, labels2 = ax_step.get_legend_handles_labels() if ax_step else ([], [])
        ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='best', ncol=2)

    for idx in range(n_plots, axes.size):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_training_loss(df):
    """Training loss line plot"""
    tag_patterns = [
        ('Train/Epoch_BCE', 'BCE Loss'),
        ('Train/Epoch_Margin', 'Margin Ranking Loss'),
    ]
    plot_epoch_lines(
        df, tag_patterns,
        title='Training Loss by Epoch',
        filename=str(OUTPUT_DIR / 'training_loss.png'),
        ylabel='Loss',
        ncols=2,
    )


def plot_test_metrics(df):
    """Test-set metric line plot"""
    tag_patterns = [
        ('Test/Epoch_Loss', 'BCE Loss'),
        ('Test/Epoch_AUC', 'AUC'),
        ('Test/Epoch_F1', 'F1 Score'),
    ]
    plot_epoch_lines(
        df, tag_patterns,
        title='Test Set Metrics by Epoch',
        filename=str(OUTPUT_DIR / 'test_metrics.png'),
        ylabel='Value',
        ncols=3,
    )


def plot_monitor_metrics(df):
    """Monitor metric line plot (Factset related)"""
    _plot_monitor_internal(df)


def _plot_monitor_internal(df):
    """Monitor metric line plot logic"""
    colors = get_model_colors()
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5), squeeze=False)
    fig.suptitle('Monitor Metrics', fontsize=20, fontweight='bold', y=0.98)

    ax = axes[0][0]
    ax_epoch = ax
    ax_step = ax.twiny()
    max_epoch = 0
    max_step = 0
    seal_lines_fq = []
    epoch_lines_fq = []

    for model in sorted(df['model'].unique()):
        for flag in DIFFICULTY_ORDER:
            subset = df[(df['model'] == model) &
                        (df['flag'] == flag) &
                        (df['tag'] == 'Monitor/Factset_Quantile')]
            if subset.empty:
                continue
            agg = subset.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
            agg = agg.sort_values('epoch')
            color = colors.get((model, flag), 'gray')
            label = f"{MODEL_DISPLAY.get(model, model)}-{FLAG_SHORT[flag]}"

            if model == 'seal':
                max_step = max(max_step, agg['epoch'].max())
                seal_lines_fq.append((agg, color, label))
            else:
                agg_ep = agg[agg['epoch'] <= 20]
                if not agg_ep.empty:
                    max_epoch = max(max_epoch, agg_ep['epoch'].max())
                epoch_lines_fq.append((agg, color, label))

    if max_epoch > 0:
        ax_epoch.set_xlim(0, max_epoch)
    if max_step > 0:
        ax_step.set_xlim(0, max_step)

    for agg, color, label in epoch_lines_fq:
        ax_epoch.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.5,
                label=label, alpha=0.9)
        if len(agg) > 1 and agg['std'].notna().any():
            ax_epoch.fill_between(agg['epoch'],
                            agg['mean'] - agg['std'],
                            agg['mean'] + agg['std'],
                            color=color, alpha=0.15)

    for agg, color, label in seal_lines_fq:
        ax_step.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.8,
                label=label, alpha=0.9, linestyle='--')
        if len(agg) > 1 and agg['std'].notna().any():
            ax_step.fill_between(agg['epoch'],
                            agg['mean'] - agg['std'],
                            agg['mean'] + agg['std'],
                            color=color, alpha=0.08)

    ax_epoch.set_xlabel('Epoch', fontsize=11)
    ax_step.set_xlabel('Step (SEAL only)', fontsize=11)
    ax_epoch.set_ylabel('Value', fontsize=11)
    ax_epoch.tick_params(labelsize=9)
    ax_step.tick_params(labelsize=9)
    ax.set_title('Factset Quantile', fontsize=14)
    ax.grid(True, alpha=0.3)
    handles1, labels1 = ax_epoch.get_legend_handles_labels()
    handles2, labels2 = ax_step.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='best', ncol=2)

    wasserstein_tags = [
        ('Monitor/Wasserstein_Diff', 'Wasserstein Diff (Neg-Pos)'),
        ('Monitor/Wasserstein_Factset_vs_Neg', 'Wasserstein(Factset, Neg)'),
    ]

    for idx, (tag, title) in enumerate(wasserstein_tags, start=1):
        ax = axes[0][idx]
        ax_epoch = ax
        ax_step = ax.twiny()
        max_epoch = 0
        max_step = 0
        seal_lines = []
        epoch_lines = []

        for model in sorted(df['model'].unique()):
            for flag in DIFFICULTY_ORDER:
                subset = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['tag'] == tag)]
                if subset.empty:
                    continue
                agg = subset.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
                agg = agg.sort_values('epoch')
                color = colors.get((model, flag), 'gray')
                label = f"{MODEL_DISPLAY.get(model, model)}-{FLAG_SHORT[flag]}"

                if model == 'seal':
                    max_step = max(max_step, agg['epoch'].max())
                    seal_lines.append((agg, color, label))
                else:
                    agg_ep = agg[agg['epoch'] <= 40]
                    if not agg_ep.empty:
                        max_epoch = max(max_epoch, agg_ep['epoch'].max())
                    epoch_lines.append((agg, color, label))

        if max_epoch > 0:
            ax_epoch.set_xlim(0, max_epoch)
        if max_step > 0:
            ax_step.set_xlim(0, max_step)

        for agg, color, label in epoch_lines:
            ax_epoch.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.5,
                          label=label, alpha=0.9)
            if len(agg) > 1 and agg['std'].notna().any():
                ax_epoch.fill_between(agg['epoch'],
                                      agg['mean'] - agg['std'],
                                      agg['mean'] + agg['std'],
                                      color=color, alpha=0.15)

        for agg, color, label in seal_lines:
            ax_step.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.8,
                         label=label, alpha=0.9, linestyle='--')
            if len(agg) > 1 and agg['std'].notna().any():
                ax_step.fill_between(agg['epoch'],
                                     agg['mean'] - agg['std'],
                                     agg['mean'] + agg['std'],
                                     color=color, alpha=0.08)

        ax_epoch.set_xlabel('Epoch', fontsize=11)
        ax_step.set_xlabel('Step (SEAL only)', fontsize=11)
        ax_epoch.set_ylabel('Value', fontsize=11)
        ax_epoch.tick_params(labelsize=9)
        ax_step.tick_params(labelsize=9)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax_step.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8, loc='best', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = OUTPUT_DIR / 'monitor_metrics.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def _build_unified_legend(fig, df):
    """
    Build a unified 4x5 outer legend: 5 models x 4 negative-sampling flags,
    SEAL uses dashed lines for the step axis, the rest solid lines for the epoch axis.
    """
    colors = get_model_colors()
    handles, labels = [], []
    for model in sorted(df['model'].unique()):
        if model not in MODEL_DISPLAY:
            continue
        for flag in DIFFICULTY_ORDER:
            if flag not in FLAG_SHORT:
                continue
            color = colors.get((model, flag), 'gray')
            label = f"{MODEL_DISPLAY.get(model, model)}-{FLAG_SHORT[flag]}"
            ls = '--' if model == 'seal' else '-'
            handles.append(Line2D([0], [0], color=color, linewidth=2, linestyle=ls))
            labels.append(label)

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
               ncol=5, fontsize=8.5, frameon=True, fancybox=True,
               title='')
    return handles, labels


def plot_combined_test_monitor(df):
    """
    Merge test_metrics and monitor_metrics into a 3x2 panel figure.
    SEAL data is recorded per batch step and drawn on the top x-axis (dashed);
    the other models are drawn per epoch on the bottom x-axis (solid).
    A unified legend is arranged 4x5 at the outer bottom; each subplot no longer draws its own legend.
    The x-axis names (Epoch / Step) are placed on the right to avoid clashing with subplot titles.
    """
    colors = get_model_colors()
    seal_present = 'seal' in df['model'].unique()

    # 6 panels: (tag, title, epoch_limit_for_non_seal)
    panels = [
        ('Test/Epoch_Loss',             'Test BCE Loss',               20),
        ('Test/Epoch_AUC',              'Test AUC',                    20),
        ('Test/Epoch_F1',               'Test F1 Score',               20),
        ('Monitor/Factset_Quantile',    'Factset Quantile',            20),
        ('Monitor/Wasserstein_Diff',            'Wasserstein Diff (Neg\u2212Pos)',  40),
        ('Monitor/Wasserstein_Factset_vs_Neg',  'Wasserstein(Factset, Neg)',        40),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('Test Metrics & Monitor Metrics', fontsize=20, fontweight='bold', y=0.99)

    for idx, (tag, title, epoch_limit) in enumerate(panels):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]
        ax_epoch = ax
        ax_step = ax.twiny() if seal_present else None

        max_epoch = 0
        max_step = 0
        seal_lines = []
        epoch_lines = []

        for model in sorted(df['model'].unique()):
            if model not in MODEL_DISPLAY:
                continue
            for flag in DIFFICULTY_ORDER:
                subset = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['tag'] == tag)]
                if subset.empty:
                    continue
                agg = subset.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
                agg = agg.sort_values('epoch')
                color = colors.get((model, flag), 'gray')
                label = f"{MODEL_DISPLAY.get(model, model)}-{FLAG_SHORT[flag]}"

                if model == 'seal':
                    max_step = max(max_step, agg['epoch'].max())
                    seal_lines.append((agg, color, label))
                else:
                    agg_ep = agg[agg['epoch'] <= epoch_limit]
                    if not agg_ep.empty:
                        max_epoch = max(max_epoch, agg_ep['epoch'].max())
                    epoch_lines.append((agg, color, label))

        if max_epoch > 0:
            ax_epoch.set_xlim(0, max_epoch)
        if max_step > 0 and ax_step:
            ax_step.set_xlim(0, max_step)

        # Non-SEAL: epoch axis, solid lines
        for agg, color, label in epoch_lines:
            ax_epoch.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.5,
                          alpha=0.9)
            if len(agg) > 1 and agg['std'].notna().any():
                ax_epoch.fill_between(agg['epoch'],
                                      agg['mean'] - agg['std'],
                                      agg['mean'] + agg['std'],
                                      color=color, alpha=0.15)

        # SEAL: step axis, dashed lines
        if ax_step:
            for agg, color, label in seal_lines:
                ax_step.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.8,
                             alpha=0.9, linestyle='--')
                if len(agg) > 1 and agg['std'].notna().any():
                    ax_step.fill_between(agg['epoch'],
                                         agg['mean'] - agg['std'],
                                         agg['mean'] + agg['std'],
                                         color=color, alpha=0.08)

        # Place x-axis name on the right
        ax_epoch.set_xlabel('Epoch', fontsize=11, loc='right')
        if ax_step:
            ax_step.set_xlabel('Step (SEAL only)', fontsize=11, loc='right')
        ax_epoch.set_ylabel('Value', fontsize=11)
        ax_epoch.tick_params(labelsize=9)
        if ax_step:
            ax_step.tick_params(labelsize=9)

        ax.set_title(title, fontsize=13, pad=8)
        ax.grid(True, alpha=0.3)

    _build_unified_legend(fig, df)

    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    fname = OUTPUT_DIR / 'test_monitor_combined.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_per_model_comparison(df):
    """One figure per model, comparing training loss and Test AUC across the 4 parameter configs"""
    colors = get_model_colors()

    for model in sorted(df['model'].unique()):
        mt = MODEL_METRIC_TAGS.get(model, MODEL_METRIC_TAGS['bigru'])
        tag_list = [
            (mt.get('train_loss', 'Train/Epoch_BCE'), METRIC_DISPLAY['train_loss']),
            (mt.get('test_auc', 'Test/Epoch_AUC'), METRIC_DISPLAY['test_auc']),
            (mt.get('test_f1', 'Test/Epoch_F1'), METRIC_DISPLAY['test_f1']),
        ]
        valid_tags = [(t, d) for t, d in tag_list
                      if t and not df[(df['model'] == model) & (df['tag'] == t)].empty]
        if not valid_tags:
            model_tags = [t for t in df[df['model'] == model]['tag'].unique()
                          if '/Epoch_' in t and not '/Batch_' in t
                          and not 'Periodic' in t and t != 'LR']
            valid_tags = [(t, t.split('/')[-1]) for t in sorted(model_tags)[:4]]

        n_plots = len(valid_tags)
        if n_plots == 0:
            continue
        ncols = min(n_plots, 3)
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows),
                                 squeeze=False)
        fig.suptitle(f'{MODEL_DISPLAY.get(model, model)}: Parameter Comparison',
                     fontsize=14, fontweight='bold')

        model_max_epoch = 0
        for tag, _ in valid_tags:
            for flag in DIFFICULTY_ORDER:
                sub = df[(df['model'] == model) & (df['flag'] == flag) & (df['tag'] == tag)]
                if not sub.empty:
                    model_max_epoch = max(model_max_epoch, sub['epoch'].max())
        if model == 'seal':
            xlim_right = model_max_epoch if model_max_epoch > 0 else 1000
        else:
            xlim_right = min(model_max_epoch, 20) if model_max_epoch > 0 else 20

        for idx, (tag, title) in enumerate(valid_tags):
            ax = axes[idx // ncols][idx % ncols]
            for flag in DIFFICULTY_ORDER:
                subset = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['tag'] == tag)]
                if subset.empty:
                    continue
                agg = subset.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
                agg = agg.sort_values('epoch')
                color = colors.get((model, flag), 'gray')
                label = FLAG_SHORT[flag]
                ax.plot(agg['epoch'], agg['mean'], color=color, linewidth=2,
                        label=label)
                if agg['std'].notna().any():
                    ax.fill_between(agg['epoch'],
                                    agg['mean'] - agg['std'],
                                    agg['mean'] + agg['std'],
                                    color=color, alpha=0.12)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_xlim(left=0, right=xlim_right)
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(title='Neg Sample Config', fontsize=8, title_fontsize=9)

        for idx in range(n_plots, axes.size):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fname = OUTPUT_DIR / f'per_model_{model}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")


def plot_summary_table(best_df, epoch_df, err_df=None, err_label='95% CI'):
    """Create a colored summary table figure (including Wasserstein_Diff and mean epoch duration)

    err_df: optional seed aggregate from aggregate_best_metrics(). When given, the values in best_df
    are means over the repeats and every metric cell is annotated with its across-seed uncertainty
    (half-width of the err_label interval), with the number of runs stated in the title.
    """
    # Monitor/* now describes the validation split that drives early stopping, while the table
    # reports the held-out test split. Newer runs therefore log the test-split FactSet statistics
    # under MonitorTest/*; legacy runs only have Monitor/*, which is what the fallback keeps.
    all_tags = set(best_df['tag'].unique()) if not best_df.empty else set()
    if 'MonitorTest/Factset_Quantile' in all_tags:
        factset_tags = ['MonitorTest/Factset_Quantile', 'MonitorTest/Wasserstein_Diff']
    else:
        factset_tags = ['Monitor/Factset_Quantile', 'Monitor/Wasserstein_Diff']
    key_tags = [
        'Test/Epoch_AUC',
        'Test/Epoch_F1',
        'Test/Epoch_Loss',
        'Train/Epoch_BCE',
    ] + factset_tags
    available_tags = [t for t in key_tags if t in best_df['tag'].unique()]
    if not available_tags:
        return

    # Compute the mean epoch duration of each (model, flag)
    time_tag = 'Time/Epoch_Duration'
    time_per_key = {}
    if time_tag in epoch_df['tag'].unique():
        time_sub = epoch_df[epoch_df['tag'] == time_tag]
        time_per_key = time_sub.groupby(['model', 'flag'])['value'].mean().to_dict()

    # Optional across-seed uncertainty per (model, flag, tag)
    err_map = {}
    n_runs = 0
    if err_df is not None and not err_df.empty:
        err_col = 'ci95' if 'ci95' in err_df.columns else ('std' if 'std' in err_df.columns else None)
        if err_col is not None:
            for _, row in err_df.iterrows():
                if np.isfinite(row[err_col]):
                    err_map[(row['model'], row['flag'], row['tag'])] = float(row[err_col])
        if 'n' in err_df.columns and len(err_df):
            n_runs = int(np.nanmax(err_df['n'].to_numpy(dtype=float)))

    row_keys = []
    for model in sorted(best_df['model'].unique()):
        for flag in DIFFICULTY_ORDER:
            has_data = any(
                len(best_df[(best_df['model'] == model) &
                             (best_df['flag'] == flag) &
                             (best_df['tag'] == tag)]) > 0
                for tag in available_tags
            )
            if has_data:
                row_keys.append((model, flag))

    if not row_keys:
        return

    table_data = []
    row_labels = []
    has_time = len(time_per_key) > 0

    for model, flag in row_keys:
        row = []
        for tag in available_tags:
            val = best_df[(best_df['model'] == model) &
                          (best_df['flag'] == flag) &
                          (best_df['tag'] == tag)]['value']
            row.append(val.iloc[0] if len(val) > 0 else np.nan)
        if has_time:
            row.append(time_per_key.get((model, flag), np.nan))
        table_data.append(row)
        row_labels.append(f"{MODEL_DISPLAY.get(model, model)} [{flag}]")

    data_arr = np.array(table_data)
    col_labels = [t.split('/')[-1] for t in available_tags]
    if has_time:
        col_labels.append('Avg\nTime/Epoch (s)')

    n_rows, n_cols = data_arr.shape
    fig, ax = plt.subplots(figsize=(3.6 * n_cols, 0.5 * n_rows + 2.0))
    ax.axis('off')
    title = 'Best Epoch Metrics Summary'
    if err_map:
        title += f' (mean ± {err_label} over {n_runs} runs)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Normalize each metric column for coloring (the duration column is handled separately)
    cell_colors = np.zeros((n_rows, n_cols, 3))
    n_metric_cols = len(available_tags)  # excludes the duration column
    for c in range(n_metric_cols):
        col_vals = data_arr[:, c]
        valid = ~np.isnan(col_vals)
        if valid.sum() > 0:
            vmin, vmax = np.nanmin(col_vals), np.nanmax(col_vals)
            if vmax > vmin:
                norm_vals = (col_vals - vmin) / (vmax - vmin)
            else:
                norm_vals = np.full_like(col_vals, 0.5)
            if 'Loss' in col_labels[c] or 'BCE' in col_labels[c]:
                cmap = plt.get_cmap('RdYlGn_r')
            else:
                cmap = plt.get_cmap('RdYlGn')
            for r in range(n_rows):
                if not np.isnan(col_vals[r]):
                    cell_colors[r, c] = cmap(norm_vals[r])[:3]

    # Duration column handled separately, no heatmap normalization
    if has_time:
        time_col = n_cols - 1
        time_vals = data_arr[:, time_col]
        valid_t = ~np.isnan(time_vals)
        if valid_t.sum() > 0:
            vmin_t, vmax_t = np.nanmin(time_vals), np.nanmax(time_vals)
            time_cmap = plt.get_cmap('RdYlGn_r')  # shorter is better
            for r in range(n_rows):
                if not np.isnan(time_vals[r]):
                    norm_t = (time_vals[r] - vmin_t) / max(vmax_t - vmin_t, 1e-6)
                    cell_colors[r, time_col] = time_cmap(norm_t)[:3]

    def _fmt_val(v, is_time=False):
        if np.isnan(v):
            return '-'
        if is_time:
            if v >= 3600:
                return f'{v/3600:.1f}h'
            elif v >= 60:
                return f'{v/60:.1f}m'
            else:
                return f'{v:.0f}s'
        if abs(v) < 0.001:
            return f'{v:.2e}'
        elif abs(v) < 1:
            return f'{v:.4f}'
        else:
            return f'{v:.3f}'

    cell_text = []
    for r in range(n_rows):
        model, flag = row_keys[r]
        row_text = []
        for c in range(n_cols):
            is_time = has_time and c == n_cols - 1
            text = _fmt_val(data_arr[r, c], is_time=is_time)
            if not is_time and c < n_metric_cols:
                err = err_map.get((model, flag, available_tags[c]))
                if err is not None:
                    text += f"\n\u00b1{err:.3f}"
            row_text.append(text)
        cell_text.append(row_text)

    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                     cellColours=cell_colors, cellLoc='center', loc='center',
                     colColours=[(0.95, 0.95, 0.95)] * n_cols)

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Row label color
    for (r, c), cell in table.get_celld().items():
        if c == -1 and r > 0:
            data_idx = r - 1
            if data_idx < len(row_labels):
                label = row_labels[data_idx]
                for model_key, display_name in MODEL_DISPLAY.items():
                    if display_name in label:
                        cmap = plt.get_cmap(MODEL_BASE_COLORS[model_key])
                        flag_label = label.split('[')[1].rstrip(']')
                        shade = SHADE_POSITIONS[DIFFICULTY_ORDER.get(flag_label, 3)]
                        cell.set_facecolor(cmap(shade))
                        cell.set_text_props(color='white', fontweight='bold')
                        break

    plt.tight_layout()
    fname = OUTPUT_DIR / 'summary_table.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    csv_data = []
    for i, (r_label, r_data) in enumerate(zip(row_labels, table_data)):
        csv_row = {'Model_Config': r_label}
        for j, col_name in enumerate(col_labels):
            csv_row[col_name] = r_data[j]
        csv_data.append(csv_row)
    csv_df = pd.DataFrame(csv_data)
    csv_fname = OUTPUT_DIR / 'best_metrics_table.csv'
    csv_df.to_csv(csv_fname, encoding='utf-8-sig', index=False)
    print(f"  Saved: {csv_fname}")


def load_sampling_modes():
    """Read the negative-sampling switch combinations from sample_setting.yaml."""
    with open(SAMPLE_SETTING_YAML, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    return {m['name']: m for m in cfg.get('sampling_modes', [])}


def _switch_text(mode):
    """'filter_factset_neg=F, intra_industry_neg=T, use_pred_neg=F' straight from the YAML."""
    tf = lambda value: 'T' if value else 'F'
    return 'filter_factset_neg=%s, intra_industry_neg=%s, use_pred_neg=%s' % (
        tf(mode.get('filter_factset_neg', False)),
        tf(mode.get('intra_industry_neg', False)),
        tf(mode.get('use_pred_neg', False)),
    )


def _mode_desc(mode):
    """Plain-language restatement of the same three switches."""
    parts = [
        'avoidance of neg. samples on FactSet edges' if mode.get('filter_factset_neg')
        else 'no avoidance of neg. samples on FactSet edges',
        'intra-industry neg. sampling on' if mode.get('intra_industry_neg')
        else 'intra-industry neg. sampling off',
        'CSV predefined negatives ON' if mode.get('use_pred_neg')
        else 'CSV predefined negatives OFF',
    ]
    return ', '.join(parts)


def plot_config_legend():
    """Draw the parameter-configuration legend figure (switches read from sample_setting.yaml)"""
    modes = load_sampling_modes()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis('off')

    colors = get_model_colors()
    for i, flag in enumerate(FLAG_ORDER):
        mode = modes.get(flag)
        if mode is None:
            continue
        params = _switch_text(mode)
        desc = _mode_desc(mode) + FLAG_HINT.get(flag, '')
        y_pos = 0.85 - i * 0.20

        color = colors.get(('bigru', flag), '#888888')
        ax.add_patch(plt.Rectangle((0.02, y_pos - 0.04), 0.03, 0.08,
                                    facecolor=color, edgecolor='black', linewidth=1))
        ax.text(0.07, y_pos, f'[{flag}]', fontsize=11, fontweight='bold',
                verticalalignment='center')

        ax.text(0.16, y_pos, params, fontsize=10, verticalalignment='center',
                fontfamily='monospace')
        ax.text(0.16, y_pos - 0.06, desc, fontsize=10, verticalalignment='center',
                color='#333333')

    ax.text(0.02, 0.93, 'Negative-Sampling Configuration Legend', fontsize=13, fontweight='bold')
    ax.text(0.02, 0.02, 'Difficulty order: ftt > ftf > fff > tff    '
             'Shade: within a model ftt darkest -> tff lightest    '
             '(BiGRU colors shown as example)',
             fontsize=9, color='#666666')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fname = OUTPUT_DIR / 'config_legend.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def _hist_to_uniform_line(limits, counts, n_bins=50, normalize_to=5000,
                          score_range=(0.0, 1.0)):
    """
    Convert a TensorBoard exponential-bin histogram into uniform-bin line data.

    Steps: resample the cumulative distribution onto n_bins uniform bins -> normalize to the normalize_to total.
    Returns (bin_edges, bin_counts) for drawing a line with ax.step.
    """
    limits = np.asarray(limits, dtype=float)
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        edges = np.linspace(score_range[0], score_range[1], n_bins + 1)
        return edges, np.zeros(n_bins)

    left_edges = np.concatenate([[score_range[0]], limits[:-1]])
    right_edges = limits

    cum = np.concatenate([[0.0], np.cumsum(counts)])

    def _interp_cum(x):
        if x <= left_edges[0]:
            return cum[0]
        if x >= right_edges[-1]:
            return cum[-1]
        idx = int(np.searchsorted(right_edges, x, side='right'))
        L, R = left_edges[idx], right_edges[idx]
        if R - L <= 1e-12:
            return cum[idx]
        return cum[idx] + (x - L) / (R - L) * (cum[idx + 1] - cum[idx])

    edges = np.linspace(score_range[0], score_range[1], n_bins + 1)
    bin_counts = np.array([
        max(0.0, _interp_cum(edges[i + 1]) - _interp_cum(edges[i]))
        for i in range(n_bins)
    ])

    s = bin_counts.sum()
    if s > 0:
        bin_counts = bin_counts / s * normalize_to

    return edges, bin_counts


def collect_histogram_data():
    """
    Walk all runs and extract histogram data (Score Distribution).
    Keep each run's list of histogram events for later alignment with the best epoch.
    Returns dict: {(model, flag, tag): [(step, limits, counts), ...]}
    """
    hist_data = {}
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if model not in MODEL_DISPLAY:
            continue

        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            timestamp, flag = parse_run_name(run_dir.name)
            if flag is None:
                continue

            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            hist_tags = ea.Tags().get('histograms', [])

            target_tags = ['Score_Distribution/Factset',
                           'Score_Distribution/TestSet_Positive',
                           'Score_Distribution/TestSet_Negative']

            for tag in target_tags:
                if tag not in hist_tags:
                    continue
                events = ea.Histograms(tag)
                if not events:
                    continue
                events_list = []
                for ev in events:
                    hv = ev.histogram_value
                    events_list.append((
                        ev.step,
                        np.array(hv.bucket_limit),
                        np.array(hv.bucket),
                    ))
                hist_data[(model, flag, tag)] = events_list

    return hist_data


CURVE_COLORS = {
    'Score_Distribution/Factset':        '#333333',
    'Score_Distribution/TestSet_Positive': '#2ca02c',
    'Score_Distribution/TestSet_Negative': '#d62728',
}

CURVE_LABELS = {
    'Score_Distribution/Factset':        'Factset',
    'Score_Distribution/TestSet_Positive': 'Test Pos',
    'Score_Distribution/TestSet_Negative': 'Test Neg',
}

CURVE_LINESTYLES = {
    'Score_Distribution/Factset':        '--',
    'Score_Distribution/TestSet_Positive': '-',
    'Score_Distribution/TestSet_Negative': ':',
}

CURVE_ALPHAS = {
    'Score_Distribution/Factset':        0.7,
    'Score_Distribution/TestSet_Positive': 0.9,
    'Score_Distribution/TestSet_Negative': 0.9,
}


def plot_score_dist_16panel(hist_data):
    """
    5x4 grid figure: rows=models (bigru, egcn, gatgru, tna, seal), columns=configs (ftt, ftf, fff, tff).
    Each cell has 3 lines (Factset / Test Pos / Test Neg); the y-axis is count.
    Uniformly resampled to 50 bins, no smoothing, normalized to 5000.
    Outputs two figures: linear y-axis + log y-axis.
    """
    models_ordered = ['bigru', 'egcn', 'gatgru', 'tna', 'seal']
    flags_ordered = ['ftt', 'ftf', 'fff', 'tff']
    score_tags = [
        'Score_Distribution/Factset',
        'Score_Distribution/TestSet_Positive',
        'Score_Distribution/TestSet_Negative',
    ]

    n_rows, n_cols = len(models_ordered), len(flags_ordered)

    TARGET_TOTAL = 5000

    factset_tag = 'Score_Distribution/Factset'
    pos_tag = 'Score_Distribution/TestSet_Positive'
    neg_tag = 'Score_Distribution/TestSet_Negative'
    bin_width = 1.0 / 50

    def _compute_wasserstein(counts1, counts2):
        """1D Wasserstein distance between two normalized probability distributions = integral|CDF1-CDF2|dx"""
        cdf1 = np.cumsum(counts1 / counts1.sum()) if counts1.sum() > 0 else np.zeros_like(counts1)
        cdf2 = np.cumsum(counts2 / counts2.sum()) if counts2.sum() > 0 else np.zeros_like(counts2)
        return np.sum(np.abs(cdf1 - cdf2)) * bin_width

    def _draw_one_figure(log_scale):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 27), squeeze=False)
        ylabel = f'Count (normalized to {TARGET_TOTAL}, log scale)' if log_scale \
                 else f'Count (normalized to {TARGET_TOTAL})'
        scale_str = 'Log Y' if log_scale else 'Linear Y'
        fig.suptitle(f'Score Distributions at Best Epoch — Normalized to {TARGET_TOTAL} ({scale_str})',
                     fontsize=22, fontweight='bold', y=0.995)

        for r, model in enumerate(models_ordered):
            for c, flag in enumerate(flags_ordered):
                ax = axes[r][c]

                line_data = {}
                for tag in score_tags:
                    key = (model, flag, tag)
                    if key not in hist_data:
                        continue
                    _, limits, counts = hist_data[key][-1]
                    if counts.sum() == 0:
                        continue

                    edges, line_counts = _hist_to_uniform_line(
                        limits, counts, n_bins=50, normalize_to=TARGET_TOTAL
                    )
                    line_data[tag] = (edges, line_counts)

                    ax.step(edges[:-1], line_counts, where='post',
                            color=CURVE_COLORS[tag],
                            linestyle=CURVE_LINESTYLES[tag],
                            linewidth=1.2,
                            alpha=CURVE_ALPHAS[tag],
                            label=CURVE_LABELS[tag])

                # Wasserstein distance annotation
                if (factset_tag in line_data and pos_tag in line_data
                        and neg_tag in line_data):
                    f_cnts = line_data[factset_tag][1]
                    p_cnts = line_data[pos_tag][1]
                    n_cnts = line_data[neg_tag][1]

                    w_fp = _compute_wasserstein(f_cnts, p_cnts)
                    w_fn = _compute_wasserstein(f_cnts, n_cnts)
                    w_pn = _compute_wasserstein(p_cnts, n_cnts)
                    delta = w_fn - w_fp

                    ann_text = (
                        f'W(F,P)={w_fp:.4f}  W(F,N)={w_fn:.4f}'
                        f'\nW(F,N)−W(F,P)={delta:+.4f}  W(P,N)={w_pn:.4f}'
                    )
                    ax.text(0.97, 0.96, ann_text,
                            transform=ax.transAxes, fontsize=12,
                            ha='right', va='top',
                            bbox=dict(boxstyle='round', facecolor='white',
                                      edgecolor='gray', alpha=0.85, pad=0.5))

                ax.set_xlim(-0.02, 1.02)
                ax.grid(True, alpha=0.25, which='both')
                if log_scale:
                    ax.set_yscale('log')

                ax.tick_params(labelsize=9)
                if c == 0:
                    ax.set_ylabel(MODEL_DISPLAY.get(model, model), fontsize=15, fontweight='bold')
                if r == 0:
                    ax.set_title(FLAG_SHORT[flag], fontsize=14, fontweight='bold')
                if r == n_rows - 1:
                    ax.set_xlabel('Score', fontsize=12)

        # Unified legend placed below the figure
        handles = [
            Line2D([0], [0], color=CURVE_COLORS[tag],
                   linestyle=CURVE_LINESTYLES[tag],
                   linewidth=2, label=CURVE_LABELS[tag])
            for tag in score_tags
        ]
        fig.legend(handles=handles, loc='lower center', ncol=3,
                   fontsize=14, frameon=True, bbox_to_anchor=(0.5, -0.01))

        # Unified y-axis label
        fig.text(0.005, 0.5, ylabel, ha='center', va='center',
                 rotation='vertical', fontsize=16, fontweight='bold')

        plt.tight_layout(rect=(0.02, 0.03, 1, 0.98))
        suffix = 'log' if log_scale else 'linear'
        fname = OUTPUT_DIR / f'score_dist_16panel_{suffix}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

    _draw_one_figure(log_scale=True)
    _draw_one_figure(log_scale=False)


def plot_wasserstein_scatter(hist_data):
    """
    Plot 5 models x 4 configs = 20 scatter points in W(F,N)-W(F,P) vs W(P,N) space.
    Colors distinguish models, shapes distinguish parameter configs.
    W(F,N)-W(F,P) = W(F,N) - W(F,P)
    """
    models_ordered = ['bigru', 'egcn', 'gatgru', 'tna', 'seal']
    flags_ordered = ['ftt', 'ftf', 'fff', 'tff']

    factset_tag = 'Score_Distribution/Factset'
    pos_tag = 'Score_Distribution/TestSet_Positive'
    neg_tag = 'Score_Distribution/TestSet_Negative'
    bin_width = 1.0 / 50

    def _compute_wasserstein(counts1, counts2):
        cdf1 = np.cumsum(counts1 / counts1.sum()) if counts1.sum() > 0 else np.zeros_like(counts1)
        cdf2 = np.cumsum(counts2 / counts2.sum()) if counts2.sum() > 0 else np.zeros_like(counts2)
        return np.sum(np.abs(cdf1 - cdf2)) * bin_width

    points = []
    for model in models_ordered:
        for flag in flags_ordered:
            line_data = {}
            for tag in [factset_tag, pos_tag, neg_tag]:
                key = (model, flag, tag)
                if key not in hist_data:
                    continue
                _, limits, counts = hist_data[key][-1]
                if counts.sum() == 0:
                    continue
                edges, line_counts = _hist_to_uniform_line(
                    limits, counts, n_bins=50, normalize_to=5000
                )
                line_data[tag] = line_counts

            if factset_tag not in line_data or pos_tag not in line_data or neg_tag not in line_data:
                continue

            w_fp = _compute_wasserstein(line_data[factset_tag], line_data[pos_tag])
            w_fn = _compute_wasserstein(line_data[factset_tag], line_data[neg_tag])
            w_pn = _compute_wasserstein(line_data[pos_tag], line_data[neg_tag])
            delta_fn_p = w_fn - w_fp

            points.append({
                'model': model,
                'flag': flag,
                'w_pn': w_pn,
                'delta_fn_p': delta_fn_p,
                'w_fp': w_fp,
                'w_fn': w_fn,
            })

    if not points:
        print("  [WARNING] No data for Wasserstein scatter plot")
        return

    # Distinguish each model by a different shade within its colormap, decreasing in order: bigru darkest -> seal lightest
    model_scatter_shades = {
        'bigru':   0.88,
        'egcn':    0.72,
        'gatgru':  0.60,
        'tna':     0.48,
        'seal':    0.38,
    }
    model_colors = {}
    for model in models_ordered:
        cmap = plt.get_cmap(MODEL_BASE_COLORS.get(model, 'Greys'))
        model_colors[model] = cmap(model_scatter_shades.get(model, 0.7))

    flag_markers = {
        'ftt': 'o',
        'ftf': 's',
        'fff': 'D',
        'tff': '^',
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Wasserstein Scatter: W(F,N)−W(F,P) vs W(P,N)',
                 fontsize=16, fontweight='bold')

    for pt in points:
        model = pt['model']
        flag = pt['flag']
        ax.scatter(pt['w_pn'], pt['delta_fn_p'],
                   c=[model_colors[model]],
                   marker=flag_markers[flag],
                   s=180,
                   edgecolors='black',
                   linewidths=0.8,
                   zorder=5,
                   label=None)

    ax.set_xlabel('W(P,N)', fontsize=14)
    ax.set_ylabel('W(F,N)−W(F,P)', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    legend_handles_model = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[m],
               markersize=12, markeredgecolor='black', markeredgewidth=0.8,
               label=MODEL_DISPLAY.get(m, m))
        for m in models_ordered if m in model_colors
    ]
    legend1 = ax.legend(handles=legend_handles_model, title='Method (color)',
                        loc='upper left', fontsize=11, title_fontsize=12,
                        framealpha=0.9)

    legend_handles_flag = [
        Line2D([0], [0], marker=flag_markers[f], color='w',
               markerfacecolor='gray', markersize=12,
               markeredgecolor='black', markeredgewidth=0.8,
               label=FLAG_SHORT[f])
        for f in flags_ordered
    ]
    ax.add_artist(legend1)
    ax.legend(handles=legend_handles_flag, title='Neg Sample Config',
              loc='lower right', fontsize=11, title_fontsize=12,
              framealpha=0.9)

    fname = OUTPUT_DIR / 'w_pn_vs_delta_scatter.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    csv_data = pd.DataFrame(points)
    csv_data['model_display'] = csv_data['model'].map(MODEL_DISPLAY)
    csv_fname = OUTPUT_DIR / 'w_pn_vs_delta_scatter.csv'
    csv_data.to_csv(csv_fname, encoding='utf-8-sig', index=False)
    print(f"  Saved: {csv_fname}")


def main():
    args = parse_args()
    global MULTI_RUN_MODE
    MULTI_RUN_MODE = args.multi_run
    if args.seed_ci and MULTI_RUN_MODE != 'confidence':
        print(f"  [INFO] --seed-ci needs every run of a (model, flag): switching multi-run mode "
              f"'{MULTI_RUN_MODE}' -> 'confidence'")
        MULTI_RUN_MODE = 'confidence'

    print("  Temporal Network Internal Data Imputation — Results Visualization")
    if MULTI_RUN_MODE == 'confidence':
        print("  Multi-run mode: confidence — curves are mean ± std over all runs of a (model, flag)")
    else:
        print("  Multi-run mode: longest — only the longest run per (model, flag) is plotted")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/7] Reading TensorBoard event files...")
    df = collect_all_data()
    if df.empty:
        print("  [ERROR] No data found! Please check the results/ directory.")
        return
    print(f"  Read {len(df)} records covering {df['model'].nunique()} models")

    df = _filter_multi_run(df, mode=MULTI_RUN_MODE)

    epoch_df = df[~df['tag'].str.contains('/Batch_')].copy()
    print(f"  Of which epoch-level records: {len(epoch_df)}")

    print("\n[2/7] Extracting best-epoch metrics...")
    best_df = get_best_epoch_metrics(epoch_df)
    print(f"  Best-metric records: {len(best_df)}")

    err_df = None
    if args.seed_ci:
        print("\n[2b/7] Aggregating the per-run best metrics (seed ensemble)...")
        per_run_df = get_best_epoch_metrics_per_run(epoch_df)
        err_df = aggregate_best_metrics(per_run_df)
        report_seed_ci(per_run_df, err_df)
        if not err_df.empty:
            # The table then shows across-seed means instead of one run's best epoch
            best_df = err_df.copy()

    print("\n[3/7] Plotting training loss line plots...")
    plot_training_loss(epoch_df)

    print("\n[4/7] Plotting combined test-and-monitor metrics figure...")
    plot_combined_test_monitor(epoch_df)

    print("\n[5/7] Plotting per-model comparison figures...")
    plot_per_model_comparison(epoch_df)

    print("\n[6/7] Plotting summary table...")
    plot_summary_table(best_df, epoch_df, err_df=err_df)
    plot_config_legend()

    print("\n[7/7] Plotting score distributions and Wasserstein scatter...")
    hist_data = collect_histogram_data()
    plot_score_dist_16panel(hist_data)
    plot_wasserstein_scatter(hist_data)

    print(f"\n  Visualization complete! Output directory: {OUTPUT_DIR}")


def parse_args():
    """CLI: --multi-run selects the run-selection policy, --seed-ci adds the across-seed aggregate."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize the TensorBoard results under results/')
    parser.add_argument('--multi-run', choices=['longest', 'confidence'], default=MULTI_RUN_MODE,
                        help="policy when a (model, flag) has several runs (e.g. a seed ensemble): "
                             "'longest' keeps the run with the most epochs, 'confidence' keeps all "
                             "runs so every curve is drawn as mean ± std")
    parser.add_argument('--seed-ci', action='store_true',
                        help='additionally aggregate the per-run best metrics into mean ± 95%% CI '
                             '(per-run CSV + summary CSV + annotated summary table); implies '
                             "--multi-run confidence")
    return parser.parse_args()


if __name__ == '__main__':
    main()
