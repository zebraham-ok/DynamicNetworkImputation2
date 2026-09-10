# -*- coding: utf-8 -*-
"""
Four-metric figure: BCE Loss / AUC / Factset Quantile / Wasserstein Diff.
  - Reads TensorBoard events under results/
  - When a (model, flag) has multiple runs, keeps the one with the most epochs
  - SEAL recorded per batch step -> top x-axis dashed; the rest per epoch solid
Output: <figures>/fig_train_4metrics.png
      <figures>/fig_train_4metrics_check.png
Usage: python Analysis/plot_train_4metrics.py [--out-dir DIR]
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import visualize_results as vr

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_out_dir():
    """Figure output directory: --out-dir first, then env var IMPUT_FIG_DIR, finally Analysis/figures/"""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out-dir', default=None)
    known, _ = parser.parse_known_args()
    return (known.out_dir
            or os.environ.get('IMPUT_FIG_DIR')
            or os.path.join(_SCRIPT_DIR, 'figures'))


FIG_DIR = _resolve_out_dir()
OUT_FILES = [os.path.join(FIG_DIR, name) for name in
             ('fig_train_4metrics.png', 'fig_train_4metrics_check.png')]

# Display name specific to this figure
DISPLAY_OVERRIDE = {'tna': 'TNA'}


def _disp(model):
    return DISPLAY_OVERRIDE.get(model, vr.MODEL_DISPLAY.get(model, model))


PANELS = [
    ('Train/Epoch_BCE',          'BCE Loss',                 40),
    ('Test/Epoch_AUC',           'AUC',                      20),
    ('Monitor/Factset_Quantile', 'Factset Quantile',         20),
    ('Monitor/Wasserstein_Diff', 'Wasserstein Diff (Neg\u2212Pos)', 40),
]


def main():
    print("Reading TensorBoard events ...")
    df = vr.collect_all_data()
    if df.empty:
        print("[ERROR] No data found!"); return
    # Temp-SEAL[fff] uses only a single run (run dir can be set via env var IMPUT_SEAL_FFF_RUN)
    _seal_fff_run = os.environ.get('IMPUT_SEAL_FFF_RUN', '0909-1523-fff')
    df = df[(df['run_id'] == _seal_fff_run) |
            ~((df['model'] == 'seal') & (df['flag'] == 'fff'))].copy()
    df = vr._filter_multi_run(df, mode='longest')
    df = df[~df['tag'].str.contains('/Batch_')].copy()
    print(f"  {len(df)} records; models: {sorted(df['model'].unique())}")

    colors = vr.get_model_colors()
    seal_present = 'seal' in df['model'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle('BCE Loss, AUC & Monitor Metrics',
                 fontsize=16, fontweight='bold', y=0.99)

    for idx, (tag, title, epoch_limit) in enumerate(PANELS):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]
        ax_epoch = ax
        ax_step = ax.twiny() if seal_present else None

        max_epoch = 0
        max_step = 0
        seal_lines, epoch_lines = [], []

        for model in sorted(df['model'].unique()):
            if model not in vr.MODEL_DISPLAY:
                continue
            for flag in vr.DIFFICULTY_ORDER:
                subset = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['tag'] == tag)]
                if subset.empty:
                    continue
                agg = subset.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
                agg = agg.sort_values('epoch')
                color = colors.get((model, flag), 'gray')
                label = f"{_disp(model)}-{vr.FLAG_SHORT[flag]}"

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
        if max_step > 0 and ax_step is not None:
            ax_step.set_xlim(0, max_step)

        for agg, color, _label in epoch_lines:
            ax_epoch.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.5, alpha=0.9)
            if len(agg) > 1 and agg['std'].notna().any():
                ax_epoch.fill_between(agg['epoch'], agg['mean'] - agg['std'],
                                      agg['mean'] + agg['std'], color=color, alpha=0.15)
        if ax_step is not None:
            for agg, color, _label in seal_lines:
                ax_step.plot(agg['epoch'], agg['mean'], color=color, linewidth=1.8,
                             alpha=0.9, linestyle='--')
                if len(agg) > 1 and agg['std'].notna().any():
                    ax_step.fill_between(agg['epoch'], agg['mean'] - agg['std'],
                                         agg['mean'] + agg['std'], color=color, alpha=0.08)

        ax_epoch.set_xlabel('Epoch', fontsize=10, loc='right')
        if ax_step is not None:
            ax_step.set_xlabel('Step (SEAL only)', fontsize=10, loc='right')
        ax_epoch.set_ylabel('Value', fontsize=10)
        ax_epoch.tick_params(labelsize=8)
        if ax_step is not None:
            ax_step.tick_params(labelsize=8)
        ax.set_title(title, fontsize=12, pad=8)
        ax.grid(True, alpha=0.3)

    handles, labels = [], []
    for model in sorted(df['model'].unique()):
        if model not in vr.MODEL_DISPLAY:
            continue
        for flag in vr.DIFFICULTY_ORDER:
            color = colors.get((model, flag), 'gray')
            label = f"{_disp(model)}-{vr.FLAG_SHORT[flag]}"
            ls = '--' if model == 'seal' else '-'
            handles.append(Line2D([0], [0], color=color, linewidth=2, linestyle=ls))
            labels.append(label)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03),
               ncol=5, fontsize=8, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    for out in OUT_FILES:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
    plt.close()


if __name__ == '__main__':
    main()
