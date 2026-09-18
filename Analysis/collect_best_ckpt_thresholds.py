"""
Collect, for every (model x sampling mode) under a dual-mode results root, the best
checkpoint and the probability threshold that belongs to it.

Layout it understands (flat root, or the `ensemble/<model>/...` variant):

    <root>/<model>/sampling_<MMDD-HHMM>/seed<NN>/<i>_<flag>_<role>/
        best_auc_model.pth                      <- checkpoint of the best-val-AUC epoch
        best_model.pth                          <- checkpoint of the best EMA5(val AUC) epoch
        model_predictions_best_auc.npy          <- its test scores {test_predictions, neg_predictions}
        model_predictions_best_auc_thresholds.json   <- youden_j / f1_max / auc of that .npy

Rule of the pipeline (see Prediction/imputation_common.yaml -> prediction.threshold_by_model):
a threshold is only valid for the checkpoint whose TEST scores produced it, i.e. the
`best_auc` tag, and the run is picked as the ftf/fff run with the highest test AUC among
all seeds x {scratch, frozen} of that model.

Outputs one JSON (default <root>/best_ckpt_thresholds.json) holding, per model and per
sampling mode, the chosen checkpoint + threshold, the full ranking of its runs, and any
warning (e.g. a model shipped only as a .zip and therefore not scanned yet).

Usage (image env, from the opensource-revise root):

    python Analysis/collect_best_ckpt_thresholds.py --root results/四次双模式-deg
    python Analysis/collect_best_ckpt_thresholds.py --root results/四次双模式-up --criterion val_auc
    python Analysis/collect_best_ckpt_thresholds.py --root results/四次双模式-deg --flags ftf fff
"""
import os
import re
import sys
import json
import glob
import argparse
from datetime import datetime

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml ships with the project env
    yaml = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

STAGE_RE = re.compile(r'^(?P<idx>\d+)_(?P<flag>[A-Za-z]+)_(?P<role>scratch|frozen)$')
SEED_RE = re.compile(r'^seed(?P<seed>\d+)$')

CKPT_FILES = {
    'best_auc': 'best_auc_model.pth',
    'best': 'best_model.pth',
}
NPY_FILES = {
    'best_auc': 'model_predictions_best_auc.npy',
    'best': 'model_predictions_best.npy',
}
THRESHOLD_FILES = {
    'best_auc': 'model_predictions_best_auc_thresholds.json',
    'best': 'model_predictions_best_thresholds.json',
}

# Same grid as Training/utils.F1_SCAN: half-open [0.1, 0.9) with step 0.001
F1_SCAN_START, F1_SCAN_STOP, F1_SCAN_STEP = 0.1, 0.9, 0.001


# --------------------------------------------------------------------------- #
# threshold helpers
# --------------------------------------------------------------------------- #
def load_npy_scores(npy_path):
    """Load {'test_predictions': [...], 'neg_predictions': [...]} into two flat arrays."""
    data = np.load(npy_path, allow_pickle=True).item()
    pos = np.asarray(data['test_predictions'][0], dtype=float).ravel()
    neg = np.asarray(data['neg_predictions'][0], dtype=float).ravel()
    return pos, neg


def auc_score(pos, neg):
    """Rank-based (Mann-Whitney) AUC with average ranks for ties -> no sklearn needed."""
    pos = np.asarray(pos, dtype=float).ravel()
    neg = np.asarray(neg, dtype=float).ravel()
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    values = np.concatenate([pos, neg])
    uniq, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    # average rank of each distinct value (1-based)
    cum = np.cumsum(counts)
    avg_rank = cum - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    rank_sum_pos = ranks[:n_pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def youden_threshold(pos, neg):
    """Threshold maximising tpr - fpr; ties resolved towards the larger threshold."""
    pos = np.asarray(pos, dtype=float).ravel()
    neg = np.asarray(neg, dtype=float).ravel()
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float('nan'), float('nan')
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    candidates = np.unique(np.concatenate([pos, neg]))
    # tpr = P(score >= t | pos) = (# pos >= t) / n_pos
    tpr = (n_pos - np.searchsorted(pos_sorted, candidates, side='left')) / n_pos
    fpr = (n_neg - np.searchsorted(neg_sorted, candidates, side='left')) / n_neg
    j = tpr - fpr
    best = int(np.argmax(j))
    best_j = float(j[best])
    tied = np.flatnonzero(np.isclose(j, best_j))
    thr = float(candidates[tied[-1]]) if tied.size else float(candidates[best])
    return thr, best_j


def f1_max_threshold(pos, neg):
    """Threshold maximising F1 on the F1_SCAN grid (pred = score >= t), half-open grid."""
    pos = np.asarray(pos, dtype=float).ravel()
    neg = np.asarray(neg, dtype=float).ravel()
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float('nan'), float('nan')
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    grid = np.arange(F1_SCAN_START, F1_SCAN_STOP, F1_SCAN_STEP)
    tp = n_pos - np.searchsorted(pos_sorted, grid, side='left')
    fp = n_neg - np.searchsorted(neg_sorted, grid, side='left')
    fn = n_pos - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / np.where(denom > 0, denom, 1), 0.0)
    best = int(np.argmax(f1))
    return float(grid[best]), float(f1[best])


def recompute_thresholds(npy_path):
    pos, neg = load_npy_scores(npy_path)
    thr_j, j_stat = youden_threshold(pos, neg)
    thr_f1, f1_best = f1_max_threshold(pos, neg)
    return {
        'auc': auc_score(pos, neg),
        'youden_j': thr_j,
        'youden_j_stat': j_stat,
        'f1_max': thr_f1,
        'f1_at_f1max': f1_best,
        'n_pos': int(pos.size),
        'n_neg': int(neg.size),
        'source': 'recomputed-from-npy',
    }


def read_threshold_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {
        'auc': data.get('auc'),
        'youden_j': data.get('youden_j'),
        'youden_j_stat': data.get('youden_j_stat'),
        'f1_max': data.get('f1_max'),
        'f1_at_f1max': data.get('f1_at_f1max'),
        'epoch': data.get('epoch'),
        'n_pos': data.get('n_pos'),
        'n_neg': data.get('n_neg'),
        'recorded_at': data.get('recorded_at'),
        'source': os.path.basename(path),
    }
    return out


def thresholds_for_stage(stage_dir, tag, warnings, label):
    """Threshold block for one ckpt tag: prefer the run's own JSON, else recompute from npy."""
    json_path = os.path.join(stage_dir, THRESHOLD_FILES[tag])
    npy_path = os.path.join(stage_dir, NPY_FILES[tag])
    if os.path.exists(json_path):
        return read_threshold_json(json_path)
    if os.path.exists(npy_path):
        warnings.append(f'{label}: {THRESHOLD_FILES[tag]} missing, thresholds recomputed from npy')
        return recompute_thresholds(npy_path)
    return None


# --------------------------------------------------------------------------- #
# metadata helpers
# --------------------------------------------------------------------------- #
def read_seed_summary(seed_dir):
    """Return {stage_name: metrics} from the seed-level summary.yaml (best effort)."""
    path = os.path.join(seed_dir, 'summary.yaml')
    if not os.path.exists(path):
        return {}
    if yaml is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data.get('results', {}) or {}


def read_run_meta(run_dir):
    """config_name / model_name of the sampling run, from config.yaml or sample_setting.yaml."""
    meta = {}
    for name in ('config.yaml', 'sample_setting.yaml'):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path) or yaml is None:
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            continue
        run = data.get('run', {}) or {}
        merged = data.get('merged_config', {}) or {}
        meta.setdefault('config_name', run.get('config_name') or (data.get('defaults', {}) or {}).get('config'))
        meta.setdefault('model_name', run.get('model_name') or merged.get('name'))
        if meta.get('config_name') and meta.get('model_name'):
            break
    return {k: v for k, v in meta.items() if v}


def rel_to_repo(path):
    try:
        return os.path.relpath(path, REPO_ROOT).replace('\\', '/')
    except ValueError:  # different drive
        return path


def discover_stage_dirs(root):
    """All directories holding a best-val-AUC prediction file (works for flat and ensemble layouts)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        if any(f in filenames for f in NPY_FILES.values()) or any(f in filenames for f in CKPT_FILES.values()):
            found.append(dirpath)
    return sorted(found)


def build_record(stage_dir, tag, warnings):
    stage_name = os.path.basename(stage_dir)
    m = STAGE_RE.match(stage_name)
    if not m:
        return None
    seed_dir = os.path.dirname(stage_dir)
    m_seed = SEED_RE.match(os.path.basename(seed_dir))
    if not m_seed:
        return None
    run_dir = os.path.dirname(seed_dir)
    model_dir = os.path.dirname(run_dir)

    seed = int(m_seed.group('seed'))
    flag = m.group('flag').lower()
    role = m.group('role')
    stage_index = int(m.group('idx'))
    label = f'{os.path.basename(model_dir)}/{os.path.basename(run_dir)}/seed{seed}/{stage_name}'

    ckpt_path = os.path.join(stage_dir, CKPT_FILES[tag])
    if not os.path.exists(ckpt_path):
        warnings.append(f'{label}: {CKPT_FILES[tag]} missing -> run skipped')
        return None

    thresholds = thresholds_for_stage(stage_dir, tag, warnings, label)
    if thresholds is None:
        warnings.append(f'{label}: no threshold json and no npy -> thresholds unknown')
        thresholds = {}

    other_tag = 'best' if tag == 'best_auc' else 'best_auc'
    other = thresholds_for_stage(stage_dir, other_tag, [], label)
    other_ckpt = os.path.join(stage_dir, CKPT_FILES[other_tag])

    summary = read_seed_summary(seed_dir).get(stage_name, {}) or {}
    if not summary:
        warnings.append(f'{label}: no entry in seed summary.yaml (val/test metrics unknown)')

    record = {
        'seed': seed,
        'flag': flag,
        'role': role,
        'stage': stage_name,
        'stage_index': stage_index,
        # kept so that both layouts (`<root>/<model>/...` and `<root>/ensemble/<model>/...`) work
        'model_key': os.path.basename(model_dir),
        'model_dir': model_dir,
        'run_name': os.path.basename(run_dir),
        'run_dir': run_dir,
        'ckpt': os.path.abspath(ckpt_path),
        'ckpt_rel': rel_to_repo(os.path.abspath(ckpt_path)),
        'ckpt_name': CKPT_FILES[tag],
        'thresholds': thresholds,
        'youden_j': thresholds.get('youden_j'),
        'f1_max': thresholds.get('f1_max'),
        'auc_from_predictions': thresholds.get('auc'),
        'val_auc': summary.get('val_auc'),
        'test_auc': summary.get('test_auc'),
        'test_f1': summary.get('test_f1'),
        'factset_quantile': summary.get('factset_quantile'),
        'factset_quantile_test': summary.get('factset_quantile_test'),
        'best_epoch': summary.get('best_epoch'),
        'epochs_run': summary.get('epochs_run'),
        'selection_criterion': summary.get('selection_criterion'),
        'npy': rel_to_repo(os.path.join(stage_dir, NPY_FILES[tag]))
        if os.path.exists(os.path.join(stage_dir, NPY_FILES[tag])) else None,
        'other_ckpt': {
            'ckpt': os.path.abspath(other_ckpt),
            'ckpt_rel': rel_to_repo(os.path.abspath(other_ckpt)) if os.path.exists(other_ckpt) else None,
            'ckpt_name': CKPT_FILES[other_tag],
            'exists': os.path.exists(other_ckpt),
            'youden_j': (other or {}).get('youden_j'),
            'f1_max': (other or {}).get('f1_max'),
            'auc_from_predictions': (other or {}).get('auc'),
            'best_epoch': (other or {}).get('epoch'),
        },
    }
    return record


# --------------------------------------------------------------------------- #
# ranking / aggregation
# --------------------------------------------------------------------------- #
def sort_key(record, criterion):
    """Descending ranking value; missing metrics sort last (never crash on None)."""
    if criterion == 'test_auc':
        value = record.get('test_auc')
        if value is None:
            value = record.get('auc_from_predictions')
    elif criterion == 'val_auc':
        value = record.get('val_auc')
        if value is None:
            value = record.get('auc_from_predictions')
    elif criterion == 'fsq_test':
        value = record.get('factset_quantile_test')
    elif criterion == 'youden_j':
        value = record.get('youden_j')
    else:
        raise ValueError(f'unknown criterion {criterion!r}')
    return -1.0 if value is None else float(value)


def _fmt(value, width, prec=4):
    return f'{value:>{width}.{prec}f}' if isinstance(value, (int, float)) else f'{"-":>{width}}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True,
                    help='dual-mode results root, e.g. results/四次双模式-deg '
                         '(relative paths resolve against the repo root)')
    ap.add_argument('--out', default=None,
                    help='output JSON path (default: <root>/best_ckpt_thresholds.json)')
    ap.add_argument('--criterion', default='test_auc',
                    choices=['test_auc', 'val_auc', 'fsq_test', 'youden_j'],
                    help='how the best run of each model x mode is picked (default: test_auc)')
    ap.add_argument('--ckpt-tag', default='best_auc', choices=['best_auc', 'best'],
                    help='which checkpoint family to report; thresholds are only valid for '
                         'best_auc = the scores in model_predictions_best_auc.npy (default)')
    ap.add_argument('--flags', nargs='+', default=None,
                    help='restrict to these sampling modes, e.g. --flags ftf fff (default: all found)')
    args = ap.parse_args()

    root = args.root if os.path.isabs(args.root) else os.path.join(REPO_ROOT, args.root)
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f'[ERROR] root not found: {root}')
        return 1

    out_path = os.path.abspath(args.out) if args.out else os.path.join(root, 'best_ckpt_thresholds.json')

    warnings = []
    records = []
    for stage_dir in discover_stage_dirs(root):
        rec = build_record(stage_dir, args.ckpt_tag, warnings)
        if rec is not None:
            records.append(rec)

    if args.flags:
        wanted = {f.lower() for f in args.flags}
        records = [r for r in records if r['flag'] in wanted]

    # group: model -> flag -> [records]
    grouped = {}
    model_meta = {}
    model_dirs = {}
    for rec in records:
        model_key = rec['model_key']
        grouped.setdefault(model_key, {}).setdefault(rec['flag'], []).append(rec)
        model_dirs[model_key] = rec['model_dir']
        if model_key not in model_meta:
            meta = read_run_meta(rec['run_dir'])
            if meta:
                model_meta[model_key] = {'run': rec['run_name'], **meta}

    models_out = {}
    print('\n' + '=' * 118)
    print(f'{"model":<22}{"mode":<6}{"seed":>5} {"role":<8}{"valAUC":>8}{"testAUC":>9}'
          f'{"FSQtest":>9}{"youden":>9}{"f1max":>8}  {"ckpt":<52}')
    print('=' * 118)

    for model_key in sorted(grouped):
        modes = {}
        for flag in sorted(grouped[model_key]):
            runs = sorted(grouped[model_key][flag], key=lambda r: sort_key(r, args.criterion), reverse=True)
            best = runs[0]
            modes[flag] = {
                'best': {
                    'seed': best['seed'],
                    'role': best['role'],
                    'stage': best['stage'],
                    'ckpt': best['ckpt'],
                    'ckpt_rel': best['ckpt_rel'],
                    'ckpt_name': best['ckpt_name'],
                    'threshold_youden_j': best['youden_j'],
                    'threshold_f1_max': best['f1_max'],
                    'auc_from_predictions': best['auc_from_predictions'],
                    'val_auc': best['val_auc'],
                    'test_auc': best['test_auc'],
                    'test_f1': best['test_f1'],
                    'factset_quantile_test': best['factset_quantile_test'],
                    'best_epoch': best['best_epoch'],
                    'selection_criterion': best['selection_criterion'],
                    'threshold_source': (best['thresholds'] or {}).get('source'),
                },
                'ranking': [
                    {
                        'rank': i + 1,
                        'seed': r['seed'],
                        'role': r['role'],
                        'stage': r['stage'],
                        'ckpt_rel': r['ckpt_rel'],
                        'val_auc': r['val_auc'],
                        'test_auc': r['test_auc'],
                        'factset_quantile_test': r['factset_quantile_test'],
                        'youden_j': r['youden_j'],
                        'f1_max': r['f1_max'],
                        'auc_from_predictions': r['auc_from_predictions'],
                        'best_epoch': r['best_epoch'],
                        'epochs_run': r['epochs_run'],
                        'threshold_source': (r['thresholds'] or {}).get('source'),
                        'other_ckpt': r['other_ckpt'],
                    }
                    for i, r in enumerate(runs)
                ],
            }
            for i, r in enumerate(runs):
                mark = '->' if i == 0 else '  '
                print(f'{model_key:<22}{flag:<6}{r["seed"]:>5} {r["role"]:<8}'
                      f'{_fmt(r["val_auc"], 8)}{_fmt(r["test_auc"], 9)}{_fmt(r["factset_quantile_test"], 9)}'
                      f'{_fmt(r["youden_j"], 9)}{_fmt(r["f1_max"], 8)}  {mark}{r["ckpt_rel"]}')

        # models shipped only as a .zip were never scanned: surface them explicitly
        models_out[model_key] = {
            'model_dir': model_dirs[model_key].replace('\\', '/'),
            'run': (model_meta.get(model_key, {}) or {}).get('run'),
            'config_name': (model_meta.get(model_key, {}) or {}).get('config_name'),
            'model_name': (model_meta.get(model_key, {}) or {}).get('model_name'),
            'modes': modes,
        }

    # a .zip counts as already extracted when one of the scanned model directories is a
    # prefix of its name (`egcn-deg.zip` -> egcn-deg/, `egcn-smooth-0916-1147.zip` ->
    # ensemble/egcn-smooth/); scanning the root alone would miss the ensemble layout
    dir_names = {rec['model_key'] for rec in records}
    dir_names |= {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    zipped = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(root, '*.zip')))]
    pending = []
    for name in zipped:
        stem = name[:-4]
        if not any(stem == d or stem.startswith(d) for d in dir_names):
            pending.append(name)

    payload = {
        'meta': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'root': root,
            'criterion': args.criterion,
            'ckpt_tag': args.ckpt_tag,
            'ckpt_file': CKPT_FILES[args.ckpt_tag],
            'threshold_file': THRESHOLD_FILES[args.ckpt_tag],
            'n_runs_scanned': len(records),
            'models': sorted(models_out),
            'note': ('A threshold is only valid for the checkpoint whose test scores produced it '
                     '(tag=best_auc). Picking the run by test AUC replays the rule used in '
                     'Prediction/imputation_common.yaml -> prediction.threshold_by_model.'),
            'pending_zips': pending,
        },
        'models': models_out,
        'warnings': warnings,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print('=' * 118)
    print(f'runs scanned : {len(records)}')
    print(f'models       : {", ".join(sorted(models_out)) if models_out else "-"}')
    if pending:
        print(f'[WARN] not unzipped yet (skipped): {", ".join(pending)}')
    if warnings:
        print(f'[WARN] {len(warnings)} warning(s), see the JSON "warnings" field')
    print(f'written to   : {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
