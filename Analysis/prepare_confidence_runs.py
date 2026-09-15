#!/usr/bin/env python3
"""Flatten a seed-ensemble run tree into the `confidence/<model>/<ts>-<flag>-sNN/` layout.

The two trees of one dataset hold the same runs in two shapes, because the analysis scripts read
them differently:

    ensemble/<model>/sampling_<ts>/seedNN/<i>_<flag>_{scratch,frozen}/...      (as uploaded)
        -> seed_ci_summary.py (summary.yaml) and seed_ci_figures.py (npy per seed)

    confidence/<model>/<ts>-<flag>-sNN/{events.*, model_predictions_best*.npy, *_thresholds.json,
                                       thresholds_history.jsonl}
        -> visualize_results.py and plot_train_4metrics.py (one run = one directory)

Usage:
    python Analysis/prepare_confidence_runs.py <model-dir-name> [<dataset-dir>] [--zip PATH]

`<model-dir-name>` is the directory name under ensemble/, i.e. the `key` used in
model_plot_config.json. With --zip the archive is unpacked into ensemble/ first (the archive root
is expected to be the sampling_<ts>/ directory). Existing files are overwritten, so re-running
after fixing a bad upload is safe. Files under .ipynb_checkpoints/ are ignored on purpose: the
checkpointed JSONs would otherwise overwrite the real threshold reports.
"""

import argparse
import os
import shutil
import sys
import zipfile

KEEP_SUFFIXES = {'.npy', '.json', '.jsonl'}


def unpack(zip_path, ens_dir):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ens_dir)
    return ens_dir


def flatten(run_dir, conf_dir, ts):
    made = []
    for seed_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('seed')):
        seed = seed_dir.name.replace('seed', '')
        for mode_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            parts = mode_dir.name.split('_')          # e.g. 1_fff_frozen
            if len(parts) < 3:
                continue
            dest = os.path.join(conf_dir, f'{ts}-{parts[1]}-s{seed}')
            os.makedirs(dest, exist_ok=True)
            for f in sorted(mode_dir.iterdir()):
                if f.is_file() and f.suffix in KEEP_SUFFIXES:
                    shutil.copy2(f, os.path.join(dest, f.name))
            for ev in sorted(mode_dir.rglob('events.out.tfevents*')):
                if '.ipynb_checkpoints' not in ev.parts:
                    shutil.copy2(ev, os.path.join(dest, ev.name))
            made.append(dest)
    return made


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', help='model directory name (the key of model_plot_config.json)')
    ap.add_argument('dataset', nargs='?', default='四次双模式',
                    help='dataset directory under results/ (default: 四次双模式)')
    ap.add_argument('--zip', default=None, help='unpack this archive into ensemble/<model>/ first')
    ap.add_argument('--run', default=None, help='run directory to flatten (default: the only sampling_* dir)')
    args = ap.parse_args()

    root = os.path.join(repo, 'results', args.dataset)
    ens_dir = os.path.join(root, 'ensemble', args.model)
    os.makedirs(ens_dir, exist_ok=True)
    if args.zip:
        unpack(args.zip, ens_dir)
        print(f'unpacked {args.zip} -> {ens_dir}')

    if args.run:
        run_dir = args.run
    else:
        runs = sorted(p for p in os.listdir(ens_dir)
                      if p.startswith('sampling_') and os.path.isdir(os.path.join(ens_dir, p)))
        if len(runs) != 1:
            sys.exit(f'expected exactly one sampling_* run under {ens_dir}, found {runs}; pass --run')
        run_dir = os.path.join(ens_dir, runs[0])

    import pathlib
    run_dir = pathlib.Path(run_dir)
    ts = run_dir.name.replace('sampling_', '')
    conf_dir = os.path.join(root, 'confidence', args.model)
    os.makedirs(conf_dir, exist_ok=True)
    made = flatten(run_dir, conf_dir, ts)
    print(f'wrote {len(made)} runs to {conf_dir}')
    for d in made:
        print('  ', os.path.basename(d), sorted(os.listdir(d)))


if __name__ == '__main__':
    main()
