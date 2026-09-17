import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.neo4j_SPLC import Neo4jClient
from collections import defaultdict
from itertools import combinations
import numpy as np
import json
from datetime import datetime

RESTRICT_INDUSTRY = True  # whether to consider only edges consistent with industry_network.json

# The imputation model to analyse = the `model` property written on each imputed
# SupplyProductTo edge (see Prediction/get_imputation_fast.py: MERGE ... {source, model}).
# Change this constant (or set the IMPUT_SCAN_MODEL environment variable) to scan another
# imputation run; everything downstream (thresholds, meta, output filename) follows it.
MODEL = os.environ.get("IMPUT_SCAN_MODEL", "egcn_ftf_s45")
print(f"Target imputation model (edge property r.model): {MODEL}")

INDUS_NETWORK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "info", "indus_network.json"
)
SCAN_STEP = 0.02  # probability-threshold scan step
# Denser sampling in the high-probability zone (zone_start, zone_end, dense_step)
HIGH_DENSE_ZONES = [
    (0.90, 0.97, 0.005),
    (0.97, 0.999, 0.001),
]
# Source of key thresholds: threshold_results.json exported by Prediction/find_threshold.py,
# default results/bootstrap/<run>/threshold_analysis/, configurable via the IMPUT_THRESHOLD_RUN
# environment variable to specify the run directory (relative to repo root or absolute).
# The file records the probability thresholds corresponding to youden_j/f1_max/f2_max;
# if absent it is skipped and only affects the key-threshold annotations.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_threshold_run = os.environ.get(
    "IMPUT_THRESHOLD_RUN", os.path.join("results", "bootstrap", "bootstrap_run"))
_threshold_run = (_threshold_run if os.path.isabs(_threshold_run)
                  else os.path.join(_REPO_ROOT, _threshold_run))
THRESHOLD_ANALYSIS_PATH = os.path.join(
    _threshold_run, "threshold_analysis", "threshold_results.json"
)


def build_upstream_lookup(indus_network_path):
    """Build the upstream-industry lookup table {downstream industry: set([upstream industry 1, ...])}"""
    with open(indus_network_path, 'r', encoding='utf-8') as f:
        indus_network = json.load(f)
    upstream_lookup = {}
    for downstream_industry, upstream_industries in indus_network.items():
        upstream_set = set()
        for upstream_item in upstream_industries:
            if isinstance(upstream_item, list):
                upstream_set.update(upstream_item)
            else:
                upstream_set.add(upstream_item)
        upstream_lookup[downstream_industry] = upstream_set
    return upstream_lookup


def check_edge_matches_industry_network(upstream_industries, downstream_industries, upstream_lookup):
    """Check whether an edge matches the upstream/downstream relation in indus_network"""
    if isinstance(upstream_industries, str):
        upstream_industries = [upstream_industries] if upstream_industries else []
    if isinstance(downstream_industries, str):
        downstream_industries = [downstream_industries] if downstream_industries else []

    for down_ind in downstream_industries:
        if down_ind in upstream_lookup:
            allowed_upstreams = upstream_lookup[down_ind]
            for up_ind in upstream_industries:
                if up_ind in allowed_upstreams:
                    return True
    return False


neo4j_client=Neo4jClient()

# Caching mechanism: first fetch nodes already flagged merged_node=true; if none, compute and write
cached_nodes = neo4j_client.execute_query('''
    MATCH (n:EntityObj {merged_node: true})
    RETURN elementId(n) AS e_id
''')

if cached_nodes:
    merged_nodes = cached_nodes
    print(f"merged_nodes count: {len(merged_nodes)} (from cache)")
else:
    print("Computing merged_nodes for the first time; filtering nodes having both factset and semi edges...")
    merged_nodes = neo4j_client.execute_query('''
        MATCH (n:EntityObj)
        WHERE EXISTS {
            MATCH (n)-[r:SupplyProductTo {source: 'factset'}]-()
        } AND EXISTS {
            MATCH (n)-[r:SupplyProductTo {source: 'semi'}]-()
        }
        SET n.merged_node = true
        RETURN elementId(n) AS e_id
    ''')
    print(f"merged_nodes count: {len(merged_nodes)} (written to cache)")

merged_nodes_id=[e['e_id'] for e in merged_nodes]

# All years involved in merged_nodes
years_result = neo4j_client.execute_query('''
    MATCH (n:EntityObj)-[r:SupplyProductTo]-(m:EntityObj)
    WHERE elementId(n) IN $merged_nodes_id AND elementId(m) IN $merged_nodes_id
    RETURN DISTINCT r.year AS year
    ORDER BY year
''', parameters={"merged_nodes_id": merged_nodes_id})

all_years = [row['year'] for row in years_result if 2013 <= row['year'] <= 2025]
print(f"\nYears involved ({len(all_years)} years, 2013-2025): {all_years}")

# Query year by year to avoid loading all edges at once
edges_between_merged = []
if RESTRICT_INDUSTRY:
    print(f"\nIndustry restriction enabled: only edges consistent with {INDUS_NETWORK_PATH}")
    upstream_lookup = build_upstream_lookup(INDUS_NETWORK_PATH)
    total_before_filter = 0

    for year in all_years:
        year_edges = neo4j_client.execute_query('''
            MATCH (n:EntityObj)-[r:SupplyProductTo]->(m:EntityObj)
            WHERE elementId(n) IN $merged_nodes_id AND elementId(m) IN $merged_nodes_id
              AND r.year = $year
            RETURN elementId(n) AS source, elementId(m) AS target,
                   r.probability AS probability, r.source AS edge_source,
                   r.year AS year, r.model AS model,
                   n.industry_1st AS upstream_industry_1st,
                   n.industry_2nd AS upstream_industry_2nd,
                   m.industry_1st AS downstream_industry_1st,
                   m.industry_2nd AS downstream_industry_2nd
        ''', parameters={"merged_nodes_id": merged_nodes_id, "year": year})

        year_filtered = []
        for edge in year_edges:
            upstream_industries = []
            if edge['upstream_industry_1st']:
                upstream_industries.append(edge['upstream_industry_1st'])
            if edge['upstream_industry_2nd']:
                upstream_industries.append(edge['upstream_industry_2nd'])
            downstream_industries = []
            if edge['downstream_industry_1st']:
                downstream_industries.append(edge['downstream_industry_1st'])
            if edge['downstream_industry_2nd']:
                downstream_industries.append(edge['downstream_industry_2nd'])
            if check_edge_matches_industry_network(upstream_industries, downstream_industries, upstream_lookup):
                year_filtered.append(edge)

        total_before_filter += len(year_edges)
        edges_between_merged.extend(year_filtered)
        print(f"  Year {year}: {len(year_edges)} -> {len(year_filtered)} (after filtering)")

    print(f"\nTotal edges before filtering: {total_before_filter}")
    print(f"Total edges after filtering: {len(edges_between_merged)}")
    if total_before_filter > 0:
        print(f"Retention ratio: {len(edges_between_merged)/total_before_filter*100:.2f}%")
else:
    for year in all_years:
        year_edges = neo4j_client.execute_query('''
            MATCH (n:EntityObj)-[r:SupplyProductTo]-(m:EntityObj)
            WHERE elementId(n) IN $merged_nodes_id AND elementId(m) IN $merged_nodes_id
              AND r.year = $year
            RETURN elementId(n) AS source, elementId(m) AS target,
                   r.probability AS probability, r.source AS edge_source,
                   r.year AS year, r.model AS model
        ''', parameters={"merged_nodes_id": merged_nodes_id, "year": year})
        edges_between_merged.extend(year_edges)
        print(f"  Year {year}: {len(year_edges)} edges")

print(f"\nTotal edges between merged_nodes: {len(edges_between_merged)}")

# Group by edge_source (using the model field)
edges_by_source = defaultdict(list)
for edge in edges_between_merged:
    edge_source = edge.get('model') or edge.get('edge_source', 'unknown')
    edges_by_source[edge_source].append(edge)

print("\nEdge statistics per data source:")
sorted_sources = sorted([s for s in edges_by_source.keys() if s is not None])
for source in sorted_sources:
    print(f"{source}: {len(edges_by_source[source])} edges")

# Edge key mapping per data source (source, target, year) -> probability
edge_keys_by_source = {}
for source, edges in edges_by_source.items():
    edge_keys_by_source[source] = {(e['source'], e['target'], e['year']): e['probability'] for e in edges}

# Start scanning at the minimum probability of MODEL
model_edges = edge_keys_by_source.get(MODEL, {})
if not model_edges:
    print(f"\nWARNING: no edges found for model={MODEL!r}; available sources: {sorted_sources}")
    print("         set MODEL (or the IMPUT_SCAN_MODEL env var) to one of them")
if model_edges:
    min_probability = min(model_edges.values())
    min_threshold = np.floor(min_probability / SCAN_STEP) * SCAN_STEP
else:
    min_threshold = 0.0
    min_probability = 0.0

# probability threshold range: base uniform sampling + denser sampling in the high-probability zone
threshold_set = set()

for t in np.arange(min_threshold, 1.01, SCAN_STEP):
    threshold_set.add(round(float(t), 10))

for zone_start, zone_end, dense_step in HIGH_DENSE_ZONES:
    zs = max(min_threshold, zone_start)
    ze = min(1.01, zone_end)
    step_count = max(1, int((ze - zs) / dense_step))
    for t in np.linspace(zs, ze, step_count + 1):
        threshold_set.add(round(float(t), 10))

threshold_set.add(1.0)
probability_thresholds = np.array(sorted(threshold_set))

print(f"\nMinimum probability of {MODEL}: {min_probability:.4f}")
print(f"Threshold scan range: {min_threshold:.2f} to 1.00")
print(f"  Base step: {SCAN_STEP}, total threshold points: {len(probability_thresholds)}")
for zs, ze, ds in HIGH_DENSE_ZONES:
    print(f"  Dense zone [{zs}, {ze}]: step {ds}")
zone_counts = {f"[{zs},{ze}]": sum(1 for t in probability_thresholds if zs <= t <= ze)
               for zs, ze, _ in HIGH_DENSE_ZONES}
for zone, n in zone_counts.items():
    print(f"  Dense zone {zone}: {n} sample points")

# Overlap ratios relative to the target model / factset / semi, and edge count per threshold
factset_overlap_on_imput_ratios = []
semi_overlap_on_imput_ratios = []
factset_overlap_on_factset_ratios = []
semi_overlap_on_semi_ratios = []
edge_counts = []

# Total edges of factset and semi (invariant to the threshold)
factset_keys = set(edge_keys_by_source.get('factset', {}).keys())
semi_keys = set(edge_keys_by_source.get('semi', {}).keys())
factset_total = len(factset_keys)
semi_total = len(semi_keys)

print("\nProbability threshold scan analysis")

for threshold in probability_thresholds:
    # Keep MODEL edges with probability >= threshold
    model_edges = edge_keys_by_source.get(MODEL, {})
    model_keys = {key for key, prob in model_edges.items() if prob >= threshold}

    if not model_keys:
        factset_overlap_on_imput_ratios.append(0.0)
        semi_overlap_on_imput_ratios.append(0.0)
        factset_overlap_on_factset_ratios.append(0.0)
        semi_overlap_on_semi_ratios.append(0.0)
        edge_counts.append(0)
        print(f"\nThreshold {threshold:.2f}: {MODEL} edge count is 0, skipping")
        continue

    factset_overlap = model_keys & factset_keys
    factset_overlap_on_imput = len(factset_overlap) / len(model_keys)
    factset_overlap_on_factset = len(factset_overlap) / factset_total
    factset_overlap_on_imput_ratios.append(factset_overlap_on_imput)
    factset_overlap_on_factset_ratios.append(factset_overlap_on_factset)

    semi_overlap = model_keys & semi_keys
    semi_overlap_on_imput = len(semi_overlap) / len(model_keys)
    semi_overlap_on_semi = len(semi_overlap) / semi_total
    semi_overlap_on_imput_ratios.append(semi_overlap_on_imput)
    semi_overlap_on_semi_ratios.append(semi_overlap_on_semi)
    edge_counts.append(len(model_keys))

    print(f"\nThreshold {threshold:.2f}:")
    print(f"  {MODEL} edge count (prob>={threshold:.2f}): {len(model_keys)}")
    print(f"  factset overlaps: {len(factset_overlap)} | of Imput: {factset_overlap_on_imput:.2%} | of FactSet: {factset_overlap_on_factset:.2%}")
    print(f"  semi overlaps:   {len(semi_overlap)} | of Imput: {semi_overlap_on_imput:.2%} | of IC-SPLC: {semi_overlap_on_semi:.2%}")

def _interpolate_at(t, thresholds_list, values_list):
    """Linearly interpolate within the threshold list and return the estimate at t"""
    if t <= thresholds_list[0]:
        return values_list[0]
    if t >= thresholds_list[-1]:
        return values_list[-1]
    for i in range(len(thresholds_list) - 1):
        if thresholds_list[i] <= t <= thresholds_list[i + 1]:
            frac = (t - thresholds_list[i]) / (thresholds_list[i + 1] - thresholds_list[i])
            return values_list[i] + frac * (values_list[i + 1] - values_list[i])
    return None

key_thresholds_data = {}
if os.path.exists(THRESHOLD_ANALYSIS_PATH):
    with open(THRESHOLD_ANALYSIS_PATH, 'r', encoding='utf-8') as f:
        ta = json.load(f)

    pt = list(probability_thresholds)

    for name, json_key in [("youden_j", "youden_j"), ("f1_max", "f1_max"), ("f2_max", "f2_max")]:
        if json_key not in ta:
            continue
        t = ta[json_key]
        exact = any(abs(t - pt_i) < 1e-9 for pt_i in pt)
        key_thresholds_data[name] = {
            "threshold": round(float(t), 6),
            "exact_match": bool(exact),
            "factset_overlap_on_imput": round(float(_interpolate_at(t, pt, factset_overlap_on_imput_ratios)), 6),
            "semi_overlap_on_imput": round(float(_interpolate_at(t, pt, semi_overlap_on_imput_ratios)), 6),
            "factset_overlap_on_factset": round(float(_interpolate_at(t, pt, factset_overlap_on_factset_ratios)), 6),
            "semi_overlap_on_semi": round(float(_interpolate_at(t, pt, semi_overlap_on_semi_ratios)), 6),
        }

    print(f"\nReading key thresholds: {THRESHOLD_ANALYSIS_PATH}")
    for name, info in key_thresholds_data.items():
        method = "exact match" if info["exact_match"] else "linear interpolation"
        print(f"  {name}: threshold={info['threshold']:.4f} ({method})")
        print(f"    FactSet/Imput={info['factset_overlap_on_imput']:.4%}  IC-SPLC/Imput={info['semi_overlap_on_imput']:.4%}")
        print(f"    FactSet/FactSet={info['factset_overlap_on_factset']:.4%}  IC-SPLC/IC-SPLC={info['semi_overlap_on_semi']:.4%}")
else:
    print(f"\nKey-threshold file does not exist, skipping: {THRESHOLD_ANALYSIS_PATH}")

# Do a fine-grained zoom-in scan when the youden_j threshold is close to 1
zoom_scan_data = None
ZOOM_TRIGGER_THRESHOLD = 0.9  # zoom is triggered only when the youden_j threshold > this value
ZOOM_N_STEPS = 100            # number of steps within the zoom range

if 'youden_j' in key_thresholds_data:
    t_youden = key_thresholds_data['youden_j']['threshold']
    if t_youden > ZOOM_TRIGGER_THRESHOLD:
        zoom_range = 10 * (1 - t_youden)
        zoom_start = max(0.80, t_youden - zoom_range)
        zoom_step = (1.0 - zoom_start) / ZOOM_N_STEPS
        zoom_thresholds = np.arange(zoom_start, 1.0 + zoom_step * 0.5, zoom_step)

        print(f"\nZoom-in fine scan")
        print(f"  Youden's J threshold: {t_youden:.6f}")
        print(f"  Zoom range: {zoom_start:.6f} ~ 1.000000")
        print(f"  Steps: {len(zoom_thresholds)}, step size: {zoom_step:.6f}")

        zoom_factset_imp = []
        zoom_semi_imp = []
        zoom_factset_ds = []
        zoom_semi_ds = []
        zoom_edge_counts = []

        model_edges = edge_keys_by_source.get(MODEL, {})
        for threshold in zoom_thresholds:
            model_keys = {key for key, prob in model_edges.items() if prob >= threshold}
            if not model_keys:
                zoom_factset_imp.append(0.0)
                zoom_semi_imp.append(0.0)
                zoom_factset_ds.append(0.0)
                zoom_semi_ds.append(0.0)
                zoom_edge_counts.append(0)
                continue

            n_edges = len(model_keys)
            fo = len(model_keys & factset_keys)
            so = len(model_keys & semi_keys)
            zoom_factset_imp.append(fo / n_edges)
            zoom_semi_imp.append(so / n_edges)
            zoom_factset_ds.append(fo / factset_total)
            zoom_semi_ds.append(so / semi_total)
            zoom_edge_counts.append(n_edges)

        zoom_key_thresholds = {}
        pt_zoom = list(zoom_thresholds)
        for name, info in key_thresholds_data.items():
            t_val = info['threshold']
            if zoom_start <= t_val <= 1.0:
                exact = any(abs(t_val - pz) < 1e-9 for pz in pt_zoom)
                zoom_key_thresholds[name] = {
                    "threshold": round(float(t_val), 6),
                    "exact_match": bool(exact),
                    "factset_overlap_on_imput": round(float(_interpolate_at(t_val, pt_zoom, zoom_factset_imp)), 6),
                    "semi_overlap_on_imput": round(float(_interpolate_at(t_val, pt_zoom, zoom_semi_imp)), 6),
                    "factset_overlap_on_factset": round(float(_interpolate_at(t_val, pt_zoom, zoom_factset_ds)), 6),
                    "semi_overlap_on_semi": round(float(_interpolate_at(t_val, pt_zoom, zoom_semi_ds)), 6),
                }

        zoom_scan_data = {
            "zoom_start": round(float(zoom_start), 6),
            "zoom_end": 1.0,
            "zoom_step": round(float(zoom_step), 6),
            "zoom_steps": ZOOM_N_STEPS,
            "thresholds": [round(float(t), 6) for t in zoom_thresholds],
            "edge_counts": zoom_edge_counts,
            "series": {
                "factset_overlap_on_imput": [round(float(v), 6) for v in zoom_factset_imp],
                "semi_overlap_on_imput": [round(float(v), 6) for v in zoom_semi_imp],
                "factset_overlap_on_factset": [round(float(v), 6) for v in zoom_factset_ds],
                "semi_overlap_on_semi": [round(float(v), 6) for v in zoom_semi_ds],
            },
            "key_thresholds": zoom_key_thresholds,
        }

        print(f"\nZoom-in key thresholds:")
        for name, info in zoom_key_thresholds.items():
            method = "exact match" if info["exact_match"] else "linear interpolation"
            print(f"  {name}: threshold={info['threshold']:.6f} ({method})")
            print(f"    FactSet/Imput={info['factset_overlap_on_imput']:.6%}  IC-SPLC/Imput={info['semi_overlap_on_imput']:.6%}")
            print(f"    FactSet/FactSet={info['factset_overlap_on_factset']:.6%}  IC-SPLC/IC-SPLC={info['semi_overlap_on_semi']:.6%}")
    else:
        print(f"\nYouden's J threshold={t_youden:.4f} <= {ZOOM_TRIGGER_THRESHOLD}, zoom-in scan not triggered")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# One result file per MODEL so that scans of different imputation runs do not overwrite each other
JSON_PATH = os.path.join(SCRIPT_DIR, f"edge_imput_overlap_scan_results_{MODEL}.json")

scan_results = {
    "meta": {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "restrict_industry": RESTRICT_INDUSTRY,
        "scan_step": SCAN_STEP,
        "dense_zones": [{"start": zs, "end": ze, "step": ds} for zs, ze, ds in HIGH_DENSE_ZONES],
        "threshold_count": len(probability_thresholds),
        "merged_nodes_count": len(merged_nodes_id),
        "factset_total_edges": factset_total,
        "semi_total_edges": semi_total,
        "total_edges_between_merged": len(edges_between_merged),
        "source_edge_counts": {s: len(edges_by_source[s]) for s in sorted_sources},
        "threshold_analysis_path": THRESHOLD_ANALYSIS_PATH,
    },
    "thresholds": [round(float(t), 6) for t in probability_thresholds],
    "series": {
        "factset_overlap_on_imput": [round(float(v), 6) for v in factset_overlap_on_imput_ratios],
        "semi_overlap_on_imput": [round(float(v), 6) for v in semi_overlap_on_imput_ratios],
        "factset_overlap_on_factset": [round(float(v), 6) for v in factset_overlap_on_factset_ratios],
        "semi_overlap_on_semi": [round(float(v), 6) for v in semi_overlap_on_semi_ratios],
    },
    "edge_counts": [int(v) for v in edge_counts],
    "key_thresholds": key_thresholds_data,
    "zoom_scan": zoom_scan_data,
}

with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(scan_results, f, ensure_ascii=False, indent=2)

print(f"\nScan results saved to: {JSON_PATH}")
print(f"Use edge_imput_overlap_plot.py to plot independently")

# Summary report
print("\nSummary")
total_edges = sum(len(edges) for edges in edges_by_source.values())
print(f"Total data sources: {len(edges_by_source)}")
print(f"Total edges: {total_edges}")

if len(edges_by_source) >= 2:
    sources = sorted([s for s in edges_by_source.keys() if s is not None])
    pairs = list(combinations(sources, 2))
    max_overlap_pair = max(pairs, key=lambda p: len(set(edge_keys_by_source[p[0]].keys()) & set(edge_keys_by_source[p[1]].keys())))
    max_overlap_count = len(set(edge_keys_by_source[max_overlap_pair[0]].keys()) & set(edge_keys_by_source[max_overlap_pair[1]].keys()))
    print(f"Data-source pair with the most overlap: {max_overlap_pair[0]} vs {max_overlap_pair[1]} ({max_overlap_count} overlapping)")
