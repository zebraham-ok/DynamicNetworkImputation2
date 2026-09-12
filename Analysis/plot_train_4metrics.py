# -*- coding: utf-8 -*-
"""
Four-metric figure: BCE Loss / AUC / Factset Quantile / Wasserstein Diff.
  - Reads TensorBoard events under results/
  - When a (model, flag) has multiple runs, keeps the one with the most epochs
  - Temp-SEAL logs per batch step -> its BCE/AUC curves go on the top (Step) axis, dashed;
    the epoch-only metrics and all other models stay on the bottom (Epoch) axis, solid
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


# (tag, title, epoch_limit, seal_batch_tag)
# Temp-SEAL is logged per batch step, so its curves live in the /Batch_* mirrors of the epoch tags.
# Monitor/Factset_Quantile and Monitor/Wasserstein_* are only computed at epoch end (there is no
# per-batch mirror), so SEAL stays on the epoch axis for those two panels: seal_batch_tag = None.
PANELS = [
    ('Train/Epoch_BCE',          'BCE Loss',                 40, 'Train/Batch_BCE'),
    ('Test/Epoch_AUC',           'AUC',                      20, 'Test/Batch_AUC'),
    ('Monitor/Factset_Quantile', 'Factset Quantile',         20, None),
    ('Monitor/Wasserstein_Diff', 'Wasserstein Diff (Neg\u2212Pos)', 40, None),
]


def main():
    print("Reading TensorBoard events ...")
    df_all = vr.collect_all_data()
    if df_all.empty:
        print("[ERROR] No data found!"); return
    # Temp-SEAL[fff] uses only a single run (run dir can be set via env var IMPUT_SEAL_FFF_RUN)
    _seal_fff_run = os.environ.get('IMPUT_SEAL_FFF_RUN', '0909-1523-fff')
    df_all = df_all[(df_all['run_id'] == _seal_fff_run) |
                    ~((df_all['model'] == 'seal') & (df_all['flag'] == 'fff'))].copy()
    # Run selection is driven by the epoch-level tags only (batch steps would dominate the count);
    # the surviving runs are then re-expanded to their full tag set, batch mirrors included.
    df = vr._filter_multi_run(df_all[~df_all['tag'].str.contains('/Batch_')], mode='longest')
    keep_runs = df[['model', 'flag', 'run_id']].drop_duplicates()
    df = df_all.merge(keep_runs, on=['model', 'flag', 'run_id'], how='inner')
    print(f"  {len(df)} records; models: {sorted(df['model'].unique())}")

    colors = vr.get_model_colors()
    seal_present = 'seal' in df['model'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    fig.suptitle('BCE Loss, AUC & Monitor Metrics',
                 fontsize=16, fontweight='bold', y=0.99)

    for idx, (tag, title, epoch_limit, seal_batch_tag) in enumerate(PANELS):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]
        ax_epoch = ax
        ax_step = None

        max_epoch = 0
        max_step = 0
        seal_step_lines, epoch_lines = [], []

        for model in sorted(df['model'].unique()):
            if model not in vr.MODEL_DISPLAY:
                continue
            for flag in vr.DIFFICULTY_ORDER:
                use_batch = (model == 'seal' and seal_batch_tag is not None)
                tag_used = seal_batch_tag if use_batch else tag
                subset = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['tag'] == tag_used)]
                if subset.empty:
                    continue
                agg = subset.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
                agg = agg.sort_values('epoch')
                color = colors.get((model, flag), 'gray')

                if use_batch:
                    # SEAL per-batch series -> owned by the top (Step) axis
                    max_step = max(max_step, agg['epoch'].max())
                    seal_step_lines.append((agg, color))
                elif model == 'seal':
                    # SEAL epoch-level series (no per-batch mirror for this metric)
                    max_epoch = max(max_epoch, agg['epoch'].max())
                    epoch_lines.append((agg, color, True))
                else:
                    agg_ep = agg[agg['epoch'] <= epoch_limit]
                    if not agg_ep.empty:
                        max_epoch = max(max_epoch, agg_ep['epoch'].max())
                    epoch_lines.append((agg, color, False))

        if max_epoch > 0:
            ax_epoch.set_xlim(0, max_epoch)

        for agg, color, is_seal in epoch_lines:
            ax_epoch.plot(agg['epoch'], agg['mean'], color=color,
                          linewidth=1.8 if is_seal else 1.5, alpha=0.9,
                          linestyle='--' if is_seal else '-')
            if len(agg) > 1 and agg['std'].notna().any():
                ax_epoch.fill_between(agg['epoch'], agg['mean'] - agg['std'],
                                      agg['mean'] + agg['std'], color=color,
                                      alpha=0.08 if is_seal else 0.15)

        if seal_step_lines:
            ax_step = ax.twiny()
            ax_step.set_xlim(0, max_step)
            for agg, color in seal_step_lines:
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
    if seal_present and PANELS[2][3] is None:
        fig.text(0.5, 0.055,
                 'Temp-SEAL: BCE / AUC are logged per batch (top axis); the FactSet statistics are '
                 'only computed at epoch end, so those two panels use the bottom axis.',
                 ha='center', fontsize=8, color='#555555')
    for out in OUT_FILES:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
    plt.close()


if __name__ == '__main__':
    main()
