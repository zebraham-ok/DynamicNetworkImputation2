# -*- coding: utf-8 -*-
"""
Four-metric figure: BCE Loss / AUC / Factset Quantile / Wasserstein Diff.
  - Split convention (2026-09-13): the loss panel is the TRAIN split, the other three panels are the
    VALIDATION split (the split that selects the epoch), and each validation curve stars the epoch
    the selection landed on. The test split is reported in summary_table.png, not as a curve.
  - Stars only in single-run mode: with --multi-run confidence one curve is the mean over a seed
    ensemble and every seed selected a different epoch, so no star is drawn (see SELECTION_STARS).
  - Panel titles carry that split as a prefix ("Train: ..." / "Validation: ..."); the legend is laid
    out one COLUMN per model and one ROW per negative-sampling setting (settings are named by
    SETTING_LABELS — currently Intra-Indus = ftf and All-Rand = fff, the only two regimes the
    seed-ensemble root contains).
  - Reads TensorBoard events under results/
  - When a (model, flag) has multiple runs: --multi-run longest (default) keeps the run with the
    most epochs; --multi-run confidence keeps every run, so each curve is drawn as mean ± std over
    the runs — this is how a seed ensemble (run_sampling.py --repeats N) becomes a CI band
  - Temp-SEAL logs per batch step -> its BCE/AUC curves go on the top (Step) axis, dashed;
    the epoch-only metrics and all other models stay on the bottom (Epoch) axis, solid
Output: <figures>/fig_train_4metrics.png
      <figures>/fig_train_4metrics_check.png
Usage: python Analysis/plot_train_4metrics.py [--results-dir DIR] [--out-dir DIR]
                                               [--multi-run longest|confidence]
       (--results-dir defaults to $IMPUT_RESULTS_DIR or <repo>/results; --out-dir defaults to
        $IMPUT_FIG_DIR or Analysis/figures)
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


def _resolve_multi_run():
    """Run-selection policy: --multi-run first, else vr.MULTI_RUN_MODE (honours IMPUT_MULTI_RUN)."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--multi-run', choices=['longest', 'confidence'], default=None)
    known, _ = parser.parse_known_args()
    return known.multi_run or vr.MULTI_RUN_MODE


def _resolve_results_dir():
    """Results root: --results-dir first, else vr.RESULTS_DIR (already honours IMPUT_RESULTS_DIR)."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--results-dir', default=None)
    known, _ = parser.parse_known_args()
    return known.results_dir


def _resolve_model_config():
    """Plot list: --model-config first, else vr.MODEL_CONFIG (already honours $IMPUT_MODEL_CONFIG)."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model-config', default=None)
    known, _ = parser.parse_known_args()
    return known.model_config or vr.MODEL_CONFIG or None


MULTI_RUN_MODE = _resolve_multi_run()

# A star is only honest when one drawn curve = one run, i.e. when that run really selected the epoch
# the star sits on. Under 'confidence' a curve is the mean over the repeats and every repeat selected
# its own epoch on the validation split — the per-repeat choices can be far apart (measured on the
# four-seed ensemble: TNA 5/9/11/17, EGCN-H 0/2/6/10, GAT-GRU·Intra-Indus 3/6/10/10), so one star at
# their mean would claim a selection the ensemble never made. Such figures therefore carry no star.
SELECTION_STARS = MULTI_RUN_MODE != 'confidence'

# Footnote used instead of vr.SELECT_MARKER_NOTE when the stars are suppressed.
CI_NO_MARKER_NOTE = ('seed ensemble: each curve is the mean \u00b1 std over the repeats and every '
                     'repeat selects its own epoch, so no single selected epoch is marked')


FIG_DIR = _resolve_out_dir()
OUT_FILES = [os.path.join(FIG_DIR, name) for name in
             ('fig_train_4metrics.png', 'fig_train_4metrics_check.png')]

# Display name specific to this figure
DISPLAY_OVERRIDE = {'tna': 'TNA'}

# Negative-sampling setting -> legend name. Only intra_industry_neg is an effective switch in the
# sampling pipeline (filter_factset_neg never changes the drawn negatives), so the four switch codes
# collapse onto two regimes: ftf = same-industry pairs removed = Intra-Indus, fff = fully random
# negatives = All-Rand. The two remaining codes are kept labelled in case an older root is plotted.
SETTING_LABELS = {
    'ftf': 'Intra-Indus',
    'fff': 'All-Rand',
    'ftt': 'Intra-Indus+NoSup',
    'tff': 'Factset-Filter',
}
# Row order of the legend grid (top row first), i.e. the drawing order of the settings.
SETTING_ORDER = ['ftf', 'fff', 'ftt', 'tff']


def _disp(model):
    return DISPLAY_OVERRIDE.get(model, vr.MODEL_DISPLAY.get(model, model))


def _setting(flag):
    return SETTING_LABELS.get(flag, vr.FLAG_SHORT.get(flag, flag))


# (tag, title, epoch_limit, seal_batch_tag, mark_selected)
# Temp-SEAL is logged per batch step, so its curves live in the /Batch_* mirrors of the epoch tags.
# Monitor/Factset_Quantile and Monitor/Wasserstein_* are only computed at epoch end (there is no
# per-batch mirror), so SEAL stays on the epoch axis for those two panels: seal_batch_tag = None.
# mark_selected: star the epoch chosen on the validation split — on for the validation panels, off
# for the training-loss panel.
# epoch_limit: every panel stops at the shared EPOCH_AXIS_MAX (2026-09-13), so no figure in the repo
# shows more than 20 training epochs; the SEAL step axis is a batch count, not epochs, and is not capped.
PANELS = [
    ('Train/Epoch_BCE',          'Train: BCE Loss',   vr.EPOCH_AXIS_MAX, 'Train/Batch_BCE', False),
    ('Val/Epoch_AUC',            'Validation: AUC',   vr.EPOCH_AXIS_MAX, 'Val/Batch_AUC',   True),
    ('Monitor/Factset_Quantile', 'Validation: Factset Quantile',        vr.EPOCH_AXIS_MAX, None, True),
    ('Monitor/Wasserstein_Diff', 'Validation: Wasserstein Diff (Neg\u2212Pos)', vr.EPOCH_AXIS_MAX, None, True),
]


def _resolve_tag(df, model, tag, tag_used):
    """Legacy fallback: a model without a validation curve keeps its test curve for that metric.

    Only affects runs produced before the validation split became the selection split; the panel
    title then carries a "[legacy run: test curve]" tag, so a title can never claim a split the
    drawn curve does not come from.
    """
    if not df[(df['model'] == model) & (df['tag'] == tag_used)].empty:
        return tag_used
    alt = tag_used.replace('Val/', 'Test/') if tag_used.startswith('Val/') else tag
    if alt != tag_used and not df[(df['model'] == model) & (df['tag'] == alt)].empty:
        return alt
    return tag_used


def main():
    model_config = _resolve_model_config()
    if model_config:
        vr.MODEL_CONFIG = model_config
    results_dir = _resolve_results_dir()
    if results_dir:
        vr.set_paths(results_dir)   # also reloads vr.MODEL_DISPLAY from the plot config
    else:
        vr.refresh_model_display()
    print("Reading TensorBoard events ...")
    print(f"  Results root  : {vr.RESULTS_DIR}")
    print(f"  Figure folder : {FIG_DIR}")
    df_all = vr.collect_all_data()
    if df_all.empty:
        print("[ERROR] No data found!"); return
    # Temp-SEAL[fff] may have more than one log (e.g. a superseded experimental run); pin the one to
    # display with IMPUT_SEAL_FFF_RUN. The pin is only applied when that run really exists in the
    # current results root, otherwise the seal[fff] runs are left untouched (a stale default would
    # silently drop the panel).
    _seal_fff_run = os.environ.get('IMPUT_SEAL_FFF_RUN')
    if _seal_fff_run and _seal_fff_run in set(df_all['run_id']):
        df_all = df_all[(df_all['run_id'] == _seal_fff_run) |
                        ~((df_all['model'] == 'seal') & (df_all['flag'] == 'fff'))].copy()
        print(f"  seal[fff] pinned to run {_seal_fff_run}")
    elif _seal_fff_run:
        print(f"  [WARN] IMPUT_SEAL_FFF_RUN={_seal_fff_run} is not part of this results root; "
              f"keeping every seal[fff] run")
    # Run selection is driven by the epoch-level tags only (batch steps would dominate the count);
    # the surviving runs are then re-expanded to their full tag set, batch mirrors included.
    print(f"  Multi-run mode: {MULTI_RUN_MODE}"
          + (" (curves = mean ± std over all runs of a (model, flag))"
             if MULTI_RUN_MODE == 'confidence' else " (longest run per (model, flag))"))
    df = vr._filter_multi_run(df_all[~df_all['tag'].str.contains('/Batch_')],
                              mode=MULTI_RUN_MODE)
    keep_runs = df[['model', 'flag', 'run_id']].drop_duplicates()
    df = df_all.merge(keep_runs, on=['model', 'flag', 'run_id'], how='inner')
    print(f"  {len(df)} records; models: {sorted(df['model'].unique())}")

    colors = vr.get_model_colors()
    # Only fetched when the stars are drawn: in seed-ensemble mode the per-seed selections are pooled
    # into a spread, not into one epoch (see SELECTION_STARS).
    selected = vr.get_selected_epochs(df) if SELECTION_STARS else {}
    seal_present = 'seal' in df['model'].unique()
    print(f"  Selected-epoch stars: {'on' if SELECTION_STARS else 'off (seed ensemble)'}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    fig.suptitle('Training Loss & Validation Metrics',
                 fontsize=16, fontweight='bold', y=0.99)

    for idx, (tag, title, epoch_limit, seal_batch_tag, mark) in enumerate(PANELS):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]
        ax_epoch = ax
        ax_step = None

        max_epoch = 0
        max_step = 0
        seal_step_lines, epoch_lines = [], []
        mark_jobs = []          # deferred: the step axis only exists once a SEAL curve showed up
        legacy_test = False     # a pre-09-12 run with no Val curve falls back to its Test curve

        for model in sorted(df['model'].unique()):
            if model not in vr.MODEL_DISPLAY:
                continue
            for flag in vr.DIFFICULTY_ORDER:
                use_batch = (model == 'seal' and seal_batch_tag is not None)
                tag_used = _resolve_tag(df, model, tag, seal_batch_tag if use_batch else tag)
                subset = df[(df['model'] == model) &
                            (df['flag'] == flag) &
                            (df['tag'] == tag_used)]
                if subset.empty:
                    continue
                legacy_test = legacy_test or tag_used.startswith('Test/')
                agg = subset.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
                agg = agg.sort_values('epoch')
                color = colors.get((model, flag), 'gray')
                if mark and SELECTION_STARS:
                    mark_jobs.append(('step' if use_batch else 'epoch', agg, color,
                                      selected.get((model, flag))))

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

        # Stars go on after the curves so they sit on top of both the lines and the bands.
        for where, agg, color, epochs in mark_jobs:
            vr.mark_selected_point(ax_step if where == 'step' else ax_epoch,
                                   agg, color, epochs)

        ax_epoch.set_xlabel('Epoch', fontsize=10, loc='right')
        if ax_step is not None:
            ax_step.set_xlabel('Step (SEAL only)', fontsize=10, loc='right')
        ax_epoch.set_ylabel('Value', fontsize=10)
        ax_epoch.tick_params(labelsize=8)
        if ax_step is not None:
            ax_step.tick_params(labelsize=8)
        ax.set_title(title + (' [legacy run: test curve]' if legacy_test else ''),
                     fontsize=12, pad=8)
        ax.grid(True, alpha=0.3)

    models = [m for m in sorted(df['model'].unique()) if m in vr.MODEL_DISPLAY]
    settings = [f for f in SETTING_ORDER if (df['flag'] == f).any()]
    handles, labels = [], []
    for model in models:
        for flag in settings:
            # A (model, setting) cell without a run still occupies its grid slot, so that a column
            # never drifts onto the next model.
            if df[(df['model'] == model) & (df['flag'] == flag)].empty:
                handles.append(Line2D([0], [0], color='none'))
                labels.append('')
                continue
            color = colors.get((model, flag), 'gray')
            ls = '--' if model == 'seal' else '-'
            handles.append(Line2D([0], [0], color=color, linewidth=2, linestyle=ls))
            labels.append(f"{_disp(model)} \u00b7 {_setting(flag)}")
    # matplotlib fills a legend column-major, so a model-major entry list with ncol = #models puts
    # every model in its own column and every setting in its own row.
    ncol = max(len(models), 1)
    legend_rows = max(len(settings), 1)
    legend_y = -0.03 if legend_rows <= 4 else -0.03 - 0.03 * (legend_rows - 4)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, legend_y),
               ncol=ncol, fontsize=8 if ncol <= 6 else 7, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.07 if legend_rows <= 4 else 0.04 + 0.018 * legend_rows, 1, 0.96])
    if seal_present and PANELS[2][3] is None:
        fig.text(0.5, 0.052,
                 'Temp-SEAL: BCE / AUC are logged per batch (top axis); the FactSet statistics are '
                 'only computed at epoch end, so those two panels use the bottom axis.',
                 ha='center', fontsize=8, color='#555555')
    fig.text(0.5, 0.03, vr.SELECT_MARKER_NOTE if SELECTION_STARS else CI_NO_MARKER_NOTE,
             ha='center', fontsize=8, color='#555555')
    for out in OUT_FILES:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
    plt.close()


if __name__ == '__main__':
    main()
