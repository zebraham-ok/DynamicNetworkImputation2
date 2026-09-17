"""
Standalone plotting script: reads the JSON results saved by edge_imput_overlap_scan.py and produces a dual-Y-axis line plot.
No database connection needed; the chart style can be tweaked repeatedly without re-scanning.
"""
import os, json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Set CJK font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same switch as in edge_imput_overlap_scan.py: which imputation run to plot.
MODEL = os.environ.get("IMPUT_SCAN_MODEL", "egcn_ftf_s45")

JSON_PATH = os.path.join(SCRIPT_DIR, f"edge_imput_overlap_scan_results_{MODEL}.json")
if not os.path.exists(JSON_PATH):  # legacy single-model file name
    _legacy = os.path.join(SCRIPT_DIR, "edge_imput_overlap_scan_results.json")
    if os.path.exists(_legacy):
        JSON_PATH = _legacy
OUT_PNG = os.path.join(SCRIPT_DIR, f"edge_imput_overlap_ratio_by_probability_{MODEL}.png")
OUT_SVG = os.path.join(SCRIPT_DIR, f"edge_imput_overlap_ratio_by_probability_{MODEL}.svg")

# Source of key thresholds (file exported by threshold_analysis, recording the probability thresholds for youden_j / f1_max / f2_max).
# Defaults to results/bootstrap/<run>/threshold_analysis/; use the IMPUT_THRESHOLD_RUN
# environment variable to specify the run directory (relative to repo root or absolute). Falls back to local interpolation when the file is absent.
_REPO_ROOT = os.path.dirname(SCRIPT_DIR)
_threshold_run = os.environ.get(
    "IMPUT_THRESHOLD_RUN", os.path.join("results", "bootstrap", "bootstrap_run"))
_threshold_run = (_threshold_run if os.path.isabs(_threshold_run)
                  else os.path.join(_REPO_ROOT, _threshold_run))
THRESHOLD_ANALYSIS_PATH = os.path.join(
    _threshold_run, "threshold_analysis", "threshold_results.json"
)

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

meta = data["meta"]
thresholds = data["thresholds"]
series = data["series"]
edge_counts = data.get("edge_counts", [])

model = meta["model"]
print(f"Reading scan results: {JSON_PATH}")
print(f"  Model: {model}")
print(f"  Threshold range: {thresholds[0]:.2f} ~ {thresholds[-1]:.2f} ({len(thresholds)} points)")
print(f"  FactSet total edges: {meta['factset_total_edges']}")
print(f"  IC-SPLC total edges: {meta['semi_total_edges']}")
print(f"  merged_nodes: {meta['merged_nodes_count']}")

# Prefer the key thresholds computed by the scan script; otherwise interpolate locally
key_thresholds = data.get("key_thresholds", {})

if not key_thresholds and os.path.exists(THRESHOLD_ANALYSIS_PATH):
    print(f"\nNo key_thresholds in scan JSON; reading and interpolating locally: {THRESHOLD_ANALYSIS_PATH}")
    with open(THRESHOLD_ANALYSIS_PATH, 'r', encoding='utf-8') as f:
        ta = json.load(f)

    def _interpolate(t, ts, vs):
        if t <= ts[0]:
            return vs[0]
        if t >= ts[-1]:
            return vs[-1]
        for i in range(len(ts) - 1):
            if ts[i] <= t <= ts[i + 1]:
                frac = (t - ts[i]) / (ts[i + 1] - ts[i])
                return vs[i] + frac * (vs[i + 1] - vs[i])
        return None

    for name, json_key in [("youden_j", "youden_j"), ("f1_max", "f1_max"), ("f2_max", "f2_max")]:
        if json_key not in ta:
            continue
        t = ta[json_key]
        key_thresholds[name] = {
            "threshold": round(float(t), 6),
            "exact_match": False,
            "factset_overlap_on_imput": round(float(_interpolate(t, thresholds, series['factset_overlap_on_imput'])), 6),
            "semi_overlap_on_imput": round(float(_interpolate(t, thresholds, series['semi_overlap_on_imput'])), 6),
            "factset_overlap_on_factset": round(float(_interpolate(t, thresholds, series['factset_overlap_on_factset'])), 6),
            "semi_overlap_on_semi": round(float(_interpolate(t, thresholds, series['semi_overlap_on_semi'])), 6),
        }

if key_thresholds:
    print(f"\nKey thresholds ({len(key_thresholds)}):")
    for name, info in key_thresholds.items():
        method = "exact match" if info.get("exact_match") else "linear interpolation"
        print(f"  {name}: th={info['threshold']:.4f} ({method})")

# With edge_counts: 3 rows of overlap curves on top + 1 row of edge-count bars below; otherwise a single plot
has_edge_counts = bool(edge_counts) and len(edge_counts) == len(thresholds)
if has_edge_counts:
    fig = plt.figure(figsize=(14, 9))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
else:
    fig, ax1 = plt.subplots(figsize=(14, 7))

color_factset = '#1f77b4'   # blue
color_semi = '#ff7f0e'       # orange

# Left Y-axis: overlap as a fraction of the Dataset — dashed lines in the same color
line3, = ax1.plot(thresholds, series['factset_overlap_on_factset'],
                  color=color_factset, marker='', linewidth=2, linestyle='--',
                  label=f'{model}∩FactSet / FactSet')
line4, = ax1.plot(thresholds, series['semi_overlap_on_semi'],
                  color=color_semi, marker='', linewidth=2, linestyle='--',
                  label=f'{model}∩IC-SPLC / IC-SPLC')

ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Overlap / Dataset', fontsize=12, color='black')
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Right Y-axis: overlap as a fraction of Imputation — solid lines
ax2 = ax1.twinx()

line1, = ax2.plot(thresholds, series['factset_overlap_on_imput'],
                  color=color_factset, marker='', linewidth=2, linestyle='-',
                  label=f'{model}∩FactSet / {model}')
line2, = ax2.plot(thresholds, series['semi_overlap_on_imput'],
                  color=color_semi, marker='', linewidth=2, linestyle='-',
                  label=f'{model}∩IC-SPLC / {model}')

ax2.set_ylabel('Overlap / Imputation', fontsize=12, color='black')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(bottom=0)

# Key-threshold vertical lines
vline_colors = {'f1_max': '#d62728'}
vline_labels = {'f1_max': 'F1 max'}
vline_styles = {'f1_max': 'dashed'}

# X-axis range, used to decide whether thresholds are too close and need to be offset
x_range = thresholds[-1] - thresholds[0]
CLOSE_THRESHOLD = 0.02 * x_range  # a gap below this value is considered too close

# Sort by threshold before plotting to allow stagger offsets
sorted_items = sorted(
    [(name, info) for name, info in key_thresholds.items() if name in vline_colors],
    key=lambda x: x[1]['threshold']
)

for i, (name, info) in enumerate(sorted_items):
    th = info['threshold']
    vc = vline_colors[name]
    vs = vline_styles[name]
    vl = vline_labels[name]

    # Vertical line
    ax1.axvline(x=th, color=vc, linestyle=vs, linewidth=1.5, alpha=0.8, zorder=2)

    # Extract intersection values
    imp_factset = info['factset_overlap_on_imput']
    imp_semi = info['semi_overlap_on_imput']
    ds_factset = info['factset_overlap_on_factset']
    ds_semi = info['semi_overlap_on_semi']

    # Whether the gap to the previous threshold is too small; if so, alternate the offset direction
    prev_th = sorted_items[i - 1][1]['threshold'] if i > 0 else None
    too_close = prev_th is not None and (th - prev_th) < CLOSE_THRESHOLD
    stagger = (i % 2)

    ax1.annotate(f'{vl}\nth={th:.5f}',
                 xy=(th, 1.0), xycoords=('data', 'axes fraction'),
                 xytext=(0, 8), textcoords='offset points',
                 ha='center', va='bottom',
                 fontsize=10, color=vc, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor=vc, alpha=0.85),
                 zorder=6)

    # --- Left axis (Dataset) intersection annotations ---
    ds_sign = -1 if (stagger and too_close) else 1
    ds_dx, ds_dy1, ds_dy2 = 8 * ds_sign, 6, -10
    ax1.annotate(f'{ds_factset:.3%}',
                 xy=(th, ds_factset), fontsize=9, color=color_factset, fontweight='bold',
                 xytext=(ds_dx, ds_dy1), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor=color_factset, alpha=0.85),
                 zorder=6)
    ax1.annotate(f'{ds_semi:.3%}',
                 xy=(th, ds_semi), fontsize=9, color=color_semi, fontweight='bold',
                 xytext=(ds_dx, ds_dy2), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor=color_semi, alpha=0.85),
                 zorder=6)

    # --- Right axis (Imputation) intersection annotations ---
    imp_sign = 1 if (stagger and too_close) else -1
    imp_dx, imp_dy1, imp_dy2 = 8 * imp_sign, 6, -10
    ax2.annotate(f'{imp_factset:.3%}',
                 xy=(th, imp_factset), fontsize=9, color=color_factset, fontweight='bold',
                 xytext=(imp_dx, imp_dy1), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor=color_factset, alpha=0.85),
                 zorder=6)
    ax2.annotate(f'{imp_semi:.3%}',
                 xy=(th, imp_semi), fontsize=9, color=color_semi, fontweight='bold',
                 xytext=(imp_dx, imp_dy2), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                           edgecolor=color_semi, alpha=0.85),
                 zorder=6)

    print(f"  Drew {name} vertical line x={th:.4f}")

# Annotate the maximum of each curve
max_series = [
    (ax1, 'factset_overlap_on_factset', color_factset, 'FactSet', 'left', -10),
    (ax1, 'semi_overlap_on_semi', color_semi, 'IC-SPLC', 'left', -22),
    (ax2, 'factset_overlap_on_imput', color_factset, 'FactSet', 'right', -6),
    (ax2, 'semi_overlap_on_imput', color_semi, 'IC-SPLC', 'right', -18),
]

for ax, key, color, label, side, y_offset in max_series:
    vals = series[key]
    max_idx = np.argmax(vals)
    max_th = thresholds[max_idx]
    max_val = vals[max_idx]
    ha_align = 'right' if side == 'left' else 'left'
    x_offset = -6 if side == 'left' else 6
    marker = '*' if side == 'left' else '*'
    ax.plot(max_th, max_val, marker=marker, color=color, markersize=10, zorder=3)
    ax.annotate(f'{label}\n{max_val:.2%}',
                xy=(max_th, max_val), fontsize=9, color=color, fontweight='bold',
                xytext=(x_offset, y_offset), textcoords='offset points',
                ha=ha_align, va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.85),
                zorder=5)

curves = [line3, line4, line1, line2]
curve_labels = [l.get_label() for l in curves]
ax1.legend(curves, curve_labels, fontsize=9, loc='lower left',
           framealpha=0.7, ncol=2)

# Bottom: imputation edge-count bar chart
if has_edge_counts:
    ax3 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
    # Bar width taken as the minimum adjacent gap
    if len(thresholds) >= 2:
        min_gap = min(thresholds[i + 1] - thresholds[i] for i in range(len(thresholds) - 1))
    else:
        min_gap = 0.01
    bar_width = min_gap * 0.9
    ax3.bar(thresholds, edge_counts,
            width=bar_width, color='#7f7f7f', alpha=0.5,
            edgecolor='none')
    ax3.set_ylabel('N Edges\n(Imputation)', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', labelsize=9)
    ax3.tick_params(axis='y', labelsize=9)
    ax3.set_xlabel('Threshold', fontsize=12)

    for name, info in sorted_items:
        th = info['threshold']
        vc = vline_colors[name]
        vs = vline_styles[name]
        ax3.axvline(x=th, color=vc, linestyle=vs, linewidth=1.5, alpha=0.8, zorder=2)

    ax1.set_xlabel('')
    plt.setp(ax1.get_xticklabels(), visible=False)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_SVG, bbox_inches='tight')
plt.close()

print(f"\nFigures saved:")
print(f"  {OUT_PNG}")
print(f"  {OUT_SVG}")
