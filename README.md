# Dynamic Network Link Imputation for Supply Chain Reconstruction

Reference implementation of the model comparison and edge-imputation pipeline for
**temporal supply-chain network internal data completion**.

The framework trains several GNN + temporal-encoder architectures on an observed
semiconductor supply-chain network (IC-SPLC), selects the best-performing one, and
uses it to impute (score) candidate edges that are absent from the observed data.

---

## 1. Requirements

* Python 3.12
* PyTorch >= 1.11 (CUDA build recommended), PyTorch Geometric >= 2.0,
  `torch-scatter`, `torch-sparse`
* numpy, scipy, pandas, scikit-learn, networkx, matplotlib, seaborn, tensorboard, tqdm
* neo4j Python driver >= 5.0, pyyaml, python-dotenv

```bash
pip install -r requirements.txt
```

---

## 2. Data prerequisites

All code reads the network from **Neo4j** over Bolt.

Expected graph shape:

* **Nodes** — company entities, label `EntityObj`, carrying at least
  `embedding` (fixed-length numeric vector, the node feature),
  `description`, `name`, `country`, `industry_1st`, `industry_2nd`, `category_3rd`.
  `country`, `industry_2nd` and `category_3rd` are only read when the optional
  attribute one-hot is enabled (see §5.1); `industry_1st` is always read and is used solely
  to draw intra-industry negative samples, never as a model input.
* **Relationships** — directed `:SupplyProductTo`, distinguished by a `source` property with
  values such as `semi` (observed IC-SPLC edges, the training source),
  `factset` (external validation edges),
  `imputation` (edges produced by this pipeline),
  `random` / `preferential` (baseline generators).
  Each carries `year`; imputed edges additionally carry `probability` and `selected`.

Connection settings:

```bash
cp .env.example .env      # then edit NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
```

**Industry network file (required, not bundled).** Imputation and the overlap sweep both
restrict candidate pairs to the legal upstream-downstream industry pairs listed in a JSON
file of the form `{"downstream_industry": ["upstream_industry", ...]}`. Supply your own
copy at `info/indus_network.json`, or point to it elsewhere with
`--indus-network <path>` / `indus_network.path` in `Prediction/imputation_common.yaml`
(`Analysis/edge_imput_overlap_scan.py` reads the fixed path `info/indus_network.json`).

The candidate pool used at imputation time is the intersection of

* nodes having an `embedding`, and
* nodes whose `industry_1st` / `industry_2nd` appears in that industry network file.

**Data availability.** This repository contains code only. The observed IC-SPLC network,
the node embeddings and the FactSet validation edges are **not** redistributed here:
FactSet data remain subject to their commercial licence, and the observed network is
shared through the route described in the manuscript's data availability statement.

---

## 3. Usage

All commands are run **from the repository root**.

### 3.1 Train a single model

```bash
python Training/train_gatgru.py            -c gatgru_vec      # GAT-GRU
python Training/train_bigru.py             -c bigru_vec       # Node-GRU
python Training/train_egcn.py              -c egcn            # EvolveGCN-H
python Training/train_tna.py               -c tna_vec         # TNA
python Training/train_dualseal.py          -c seal            # Temp-SEAL
python Training/train_gatgru_1dire.py      -c gatgru_1dire    # unidirectional ablation
```

Each run writes to `results/<output_subdir>/<MMDD-HHMM>/`, where `<output_subdir>` is the
`output_subdir` field of the model config (e.g. `results/gatgru_vec/<MMDD-HHMM>/`), holding
checkpoints, TensorBoard events and `model_predictions_best.npy` /
`model_predictions_best_auc.npy` (test-set scores).

### 3.2 Run the negative-sampling comparison

`SampleSetting/run_sampling.py` trains the 2x2 combinations of
`filter_factset_neg` x `intra_industry_neg` (modes `ftt`, `ftf`, `fff`, `tff`)
under a shared-backbone protocol: the first mode trains the static encoder from
scratch and the remaining modes reuse it frozen.

```bash
python SampleSetting/run_sampling.py -c gatgru_vec
python SampleSetting/run_sampling.py -c gatgru_vec --dry_run      # inspect only
python SampleSetting/run_sampling.py -c bigru_vec                 # other backbones
python SampleSetting/run_sampling.py -c gatgru_vec --repeats 5    # 5 seeds (42..46)
```

Results land in `results/<model>/sampling_<timestamp>/<index>_<mode>_<scratch|frozen>/`,
summarised in `summary.yaml`. Add `--resume` to skip already-finished modes, or
`--num_rnn_layers 2` to reproduce the two-BiGRU-layer variant.

`--repeats N` runs the whole protocol N times with seeds `seed, seed+1, ...` (the first
seed comes from `--seed`, default 42). Each repeat re-initialises the model and writes its
own `seed<SEED>/` subdirectory holding its own `backbone.pth`, checkpoints, TensorBoard
events and `summary.yaml`, so differently initialised backbones can never be mixed up; the
run root then carries a `summary.yaml` with the across-seed mean, standard deviation and
95% confidence interval per metric. With the default `--repeats 1` the flat layout is
unchanged.

As soon as `--repeats > 1` the backbone donor rotates by default: the training order of the
four modes is shifted by one every repeat, so the mode trained from scratch (and handing its
backbone to the others) is a different one each time -- `ftt` in the first repeat, `ftf` in
the second, and so on. That removes the systematic advantage of whichever mode happened to be
picked as the donor. The run-root `summary.yaml` therefore reports the repeats twice: per mode
key (`aggregate`, role included) and per mode (`by_mode`, 1 scratch + N-1 frozen runs pooled,
with the two roles also kept apart). Use `--no_rotate_donor` to keep one fixed donor instead.

### 3.3 Bootstrap evaluation

```bash
python Bootstraps/bootstrap_v2_train.py    -c Bootstraps/bootstrap_v2_config.yaml
python Bootstraps/bootstrap_v2_analysis.py -d results/bootstrap/<run>/
python Bootstraps/bootstrap_highvar_analysis.py -d results/bootstrap/<run>/
```

### 3.4 Impute edges

```bash
python Prediction/get_imputation_fast.py -c gatgru_vec -p results/gatgru/<run>/best_model.pth
```

Scores every candidate pair in the industry-restricted pool and writes the selected
edges back into Neo4j with `source='imputation'`.

Threshold search:

```bash
python Prediction/find_threshold.py           -b results/bootstrap/<run>/
python Prediction/find_threshold_per_model.py -b results/bootstrap/<run>/
```

### 3.5 Evaluation and analysis

```bash
python Analysis/PrROC2_fff.py              # auto-selects results/gatgru/*fff*/model_predictions_best_auc.npy
python Analysis/visualize_results.py       # metric-vs-epoch figures
python Analysis/plot_train_4metrics.py     # training figure used in the paper
python Analysis/edge_imput_overlap_scan.py # threshold sweep vs. observed edges (queries Neo4j)
python Analysis/edge_imput_overlap_plot.py # overlap-vs-threshold curve (reads the sweep JSON)
python Analysis/seed_ci_summary.py         # across-seed mean ± 95% CI, from summary.yaml
python Analysis/seed_ci_figures.py         # per-seed ROC + mean curve with a 95% CI band
```

Which models appear in the figures, and under which name, is decided by one file per results
root — `model_plot_config.json` (see `results/四次双模式/model_plot_config.json`):

```jsonc
{ "models": [
    { "key": "egcn", "name": "EvolveGCN-H" },
 // { "key": "tna",  "name": "BiTNA" },        <- comment out one line to drop the model
]}
```

Commenting out a line (or `"include": false`) removes that model from *every* figure, since
`visualize_results.py`, `plot_train_4metrics.py`, `seed_ci_summary.py`, `seed_ci_figures.py`
and `PrROC_all_models*.py` all read the list through `Analysis/plot_models.py`; the list order
is also the legend and colour order. Directories on disk that are not listed are skipped. The
file is found next to `--results-dir` or one level above it; `--model-config PATH` overrides it.
With no such file nothing is filtered, i.e. the pre-config behaviour.

Adding a model that has just been uploaded takes two commands — unpack the archive into the
`ensemble/` tree and flatten it into the `confidence/` tree that the line-figure scripts read:

```bash
python Analysis/prepare_confidence_runs.py gatgru2layer-smooth \
       --zip "results/四次双模式/gatgru2layer-smooth-0913-1951.zip"
# then add one { "key": ..., "name": ... } line to model_plot_config.json and re-run the figures
```

`prepare_confidence_runs.py` is idempotent (re-running only refreshes the flattened copies) and
ignores the `.ipynb_checkpoints/` folders that some uploads carry.

Seed ensembles (several runs of the same model and sampling flag) are aggregated on two
levels: `Analysis/visualize_results.py --multi-run confidence` keeps every run and draws
each curve as mean ± standard deviation, `--seed-ci` additionally writes the across-run
best-epoch metrics with their 95% confidence intervals to `seed_ci_per_run.csv` /
`seed_ci_summary.csv`; `Analysis/seed_ci_summary.py` produces the same summary directly
from the per-seed `summary.yaml`, so it needs no TensorBoard event files. A run directory
may carry the seed it used as a suffix (`0912-1118-fff-s3`). Whenever the repeats rotate
their backbone donor, both scripts group by the sampling flag itself instead of by the
`scratch`/`frozen` role, because every flag plays both roles across the repeats.

`edge_imput_overlap_scan.py` writes `Analysis/edge_imput_overlap_scan_results.json`, which
`edge_imput_overlap_plot.py` reads; it is the only analysis script that opens a database
connection. The curve is exported as `edge_imput_overlap_ratio_by_probability.png`, i.e.
the overlap-vs-threshold figure of the manuscript.

Four optional environment variables point the analysis scripts at your own runs instead of
the defaults:

| Variable                | Used by                                                        | Meaning                                                                                                               |
| ----------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `IMPUT_THRESHOLD_RUN` | `edge_imput_overlap_scan.py`, `edge_imput_overlap_plot.py` | Run directory whose`threshold_analysis/threshold_results.json` supplies the marked F1-max / Youden-J thresholds     |
| `IMPUT_FFF_RUN`       | `PrROC2_fff.py`                                              | Run directory holding`model_predictions_best_auc.npy`                                                               |
| `IMPUT_FIG_DIR`       | `plot_train_4metrics.py`                                     | Output directory for the training figure (default`Analysis/figures/`; `--out-dir DIR` also works)                 |
| `IMPUT_SEAL_FFF_RUN`  | `plot_train_4metrics.py`                                     | Temp-SEAL[fff] run to plot, overriding the default (earlier runs used a sliding-window loss average and are excluded) |
| `IMPUT_MODEL_CONFIG` | `visualize_results.py`, `plot_train_4metrics.py`, `seed_ci_summary.py`, `seed_ci_figures.py`, `PrROC_all_models*.py` | Path of the `model_plot_config.json` deciding which models are drawn and how they are named (`--model-config` also works) |

The sweep tolerates a missing `threshold_results.json`: the curves are still produced, only
the annotated key thresholds are skipped. Likewise, `plot_train_4metrics.py` is the only
figure script that needs a Temp-SEAL[fff] run to exist.

---

## 4. Repository layout

```
.
├── Models/
│   ├── common.py                  # Shared layers for the non-vectorised EvolveGCN path
│   ├── common_vectorized.py       # Shared layers for the vectorised temporal models
│   ├── egcn.py                    # EvolveGCN-H (weight-evolving GCN)
│   ├── nodegru_vectorized.py      # Node-GRU (GCN + Bi-GRU)
│   ├── gatgru_vectorized.py       # GAT-GRU (GAT + Bi-GRU)      <- main model
│   ├── gatgru_1dire.py            # GAT-GRU unidirectional-GRU ablation
│   ├── grutna_vectorized.py       # TNA (temporal neighbourhood aggregation)
│   ├── tempseal.py                # Temp-SEAL (subgraph + temporal decay)
│   └── configs/                   # Per-model YAML (egcn, bigru_vec, gatgru_vec,
│                                  #   gatgru_1dire, tna_vec, seal)
├── Data/
│   ├── company_dataset.py         # Dataset entry: 3-way split + bootstrap / standard loaders
│   ├── graph_dataset.py           # Base primitives (CompanySupplyDataset, build_static_graph)
│   ├── neo4j_SPLC.py              # Neo4j client (credentials from .env)
│   ├── secret_manager.py          # .env loader
│   └── stopped_neg_sample.csv     # Header-only template for pre-defined negatives (see §6)
├── Training/
│   ├── config_loader.py           # Merge common_config.yaml + Models/configs/<name>.yaml
│   ├── common_config.yaml         # Shared training defaults
│   ├── trainer_common.py          # Shared training loop utilities
│   ├── train_egcn.py              # EvolveGCN-H
│   ├── train_bigru.py             # Node-GRU
│   ├── train_gatgru.py            # GAT-GRU
│   ├── train_gatgru_1dire.py      # GAT-GRU unidirectional ablation
│   ├── train_tna.py               # TNA
│   └── train_dualseal.py          # Temp-SEAL (step-level evaluation)
├── SampleSetting/
│   ├── run_sampling.py            # 2x2 sampling-mode driver (ftt/ftf/fff/tff)
│   └── sample_setting.yaml
├── Bootstraps/
│   ├── bootstrap_v2_train.py      # Bootstrap re-training (shared frozen backbone)
│   ├── bootstrap_v2_analysis.py   # Agreement / CI analysis over iterations
│   ├── bootstrap_highvar_analysis.py
│   ├── bootstrap_v2_config.yaml
│   └── common_config.yaml
├── Prediction/
│   ├── get_imputation_fast.py     # Batched candidate scoring + Neo4j write-back
│   ├── find_threshold.py          # Decision-threshold search (single model)
│   ├── find_threshold_per_model.py
│   └── imputation_common.yaml
├── Analysis/                      # Post-training evaluation and figure generation
│   ├── visualize_results.py       # TensorBoard reader shared by the plot scripts
│   ├── plot_train_4metrics.py     # Training figure reported in the paper
│   ├── edge_imput_overlap_scan.py # Threshold sweep vs. observed edges -> JSON
│   ├── edge_imput_overlap_plot.py # Overlap-vs-threshold curve (reads the sweep JSON)
│   ├── PrROC2_fff.py              # GAT-GRU[fff] ROC curve reported in the paper
│   ├── seed_ci_summary.py         # Across-seed mean ± 95% CI from per-seed summary.yaml
│   └── edgebank_baseline_results.csv   # Input baseline table (not an output)
├── utils.py                       # Shared helpers (config merge, dynamic import)
├── requirements.txt
├── .env.example                   # Copy to .env and fill in credentials
└── README.md
```

---

## 5. Models

| Model             | Config           | Implementation                   | Spatial encoder                 | Temporal encoder                   |
| ----------------- | ---------------- | -------------------------------- | ------------------------------- | ---------------------------------- |
| EvolveGCN-H       | `egcn`         | `Models/egcn.py`               | GCN with RNN-evolving weights   | GRU cells driving the GCN weights  |
| Node-GRU          | `bigru_vec`    | `Models/nodegru_vectorized.py` | 2-layer GCN                     | stacked Bi-GRU over node sequences |
| Temp-SEAL         | `seal`         | `Models/tempseal.py`           | SEAL-style 2-hop subgraph + GCN | temporal decay on edge weights     |
| **GAT-GRU** | `gatgru_vec`   | `Models/gatgru_vectorized.py`  | multi-head GAT                  | 1-layer Bi-GRU                     |
| GAT-GRU†         | `gatgru_1dire` | `Models/gatgru_1dire.py`       | multi-head GAT                  | unidirectional GRU (ablation)      |
| TNA               | `tna_vec`      | `Models/grutna_vectorized.py`  | 2-layer GCN                     | temporal neighbourhood aggregation |

Exact layer counts, heads, hidden sizes, dropout, batch size and learning rate are in
`Models/configs/*.yaml`; shared defaults are in `Training/common_config.yaml`.
The 2-temporal-layer variant GAT-GRU\* is obtained via
`SampleSetting/sample_setting.yaml → model_overrides.num_rnn_layers`.

### 5.1 Node features and the shared feature extractor

The node feature matrix `X` is assembled in exactly one place
(`Data/graph_dataset.py → assemble_node_features`, reused by
`Data/company_dataset.py → build_bootstrap_static_graph`) so that every stage of the
pipeline — training, negative-sampling sweep, bootstrap, imputation — sees the same layout:

```
X = [ dense text embedding (128) | one-hot(country) | one-hot(industry_2nd) | one-hot(category_3rd) ]
```

The optional attribute blocks are controlled by a single switch in the dataset section of
`Training/common_config.yaml` (and `Bootstraps/bootstrap_v2_config.yaml`):

```yaml
dataset:
  use_attr_onehot: true       # false -> use the 128-dim embedding alone
  onehot_attrs:               # Neo4j property names to encode
    - country
    - industry_2nd
    - category_3rd
```

The vocabulary of each attribute is derived from the data at load time and sorted, so no
vocabulary file has to be shipped; each block is exactly as wide as the number of distinct
non-empty values, and a node whose attribute is missing contributes an **all-zero row** for
that block (no `<NA>` category is reserved). Attribute names must match the Neo4j schema
actually being read (anonymised releases rename some properties). When the switch is off,
`X` is the 128-dimensional embedding alone.

#### 5.1.1 Optional degree channel

An **optional** extra column `log(1 + degree)` can be appended at the end of `X`
(`701 → 702` with the attributes above). This is the only place where a structural property of
the graph enters the input representation: by default the encoders use degree-normalised
(mean-type) aggregation and therefore cannot express how many neighbours a node has, only the
composition of its neighbourhood.

```yaml
dataset:
  use_attr_degree: false        # true -> append log1p(degree) as the last column of X
  degree_scope: train_only      # train_only | full | calibrated_full
  degree_property: degree       # DEPRECATED: no longer read/written
```

`degree` counts the incident `(source, target, year)` triples on both endpoints (the same
quantity `min_degree` thresholds on, restricted by `source_filter`). It is a **run-time derived
feature**: the column is recomputed on every run and is never stored in Neo4j. `degree_scope`
declares which edges the count may see:

| scope | counted from | when to use |
|---|---|---|
| `train_only` | the 80% train pool (default) | every evaluation run - train/val/test share that one column |
| `full` | every positive of the window | **deployment** (imputation) only - nothing is held out there |
| `calibrated_full` | `full`, quantile-mapped onto the training-time reference | deploying a `train_only`-trained checkpoint |

Two caveats: (i) the quantity is the one `min_degree` thresholds on, so a **whole-window** count
encodes the positive-inclusion rule itself and must never be used to score val/test - use the
channel to test *whether* degree information helps, not to claim a better model, and never put
such a run next to the no-degree baseline; (ii) turning it on changes the width of `X`, so every
checkpoint has to be retrained. See `TechnicalGuide.md` §3.5 for the full rationale, the
train → deploy drift (≈ Binomial(d, 0.8), relative sd ≈ 0.5/√d) and how to calibrate around it.

All five backbones then feed `X` through **one shared extractor**
(`Models/common.py → NodeFeatureExtractor`, built only via `build_feature_extractor`) whose
configuration is declared **once** for every backbone — in `Training/common_config.yaml`,
inherited by each `Models/configs/{model}.yaml` through the deep merge (a model that needs a
different extractor re-declares the same key and wins):

```yaml
# Training/common_config.yaml
model:
  kwargs:
    use_fc_embedding: true    # enable the shared extractor (identical for every backbone)
    fc_embed_dim: 128         # output width handed to the spatial encoder
    fc_hidden_dim: 256
    fc_num_layers: 3
```

These values equal the `NodeFeatureExtractor` defaults in `Models/common.py`
(`fc_hidden_dim: null → fc_embed_dim * 2`), and every consumer now resolves the model config
through `Training/config_loader.load_config`, so no entry point can pick up a different
extractor than the one the checkpoint was trained with.

GAT-GRU, GAT-GRU†, Node-GRU, TNA, EvolveGCN and Temp-SEAL all project `X` through this
module before their own spatial encoder, so the comparison between backbones differs only
in the encoder, not in the input representation. The projection is a genuine dimensionality
reduction once `use_attr_onehot` is enabled (`128 + attribute-columns → 128`), whereas with the one-hot
off it is a 128 → 128 re-encoder. The extractor belongs to the frozen backbone: it is
saved, reloaded and frozen together with `static_encoder` / `static_gnn` in the
shared-backbone and bootstrap protocols.

> **Note** — enabling the extractor changes the model architecture, so previously trained
> checkpoints are not compatible and the models have to be retrained.

## 6. Citation

```bibtex
@article{dynamicNetworkLinkImputation2026,
  title={Dynamic Network Link Imputation for Supply Chain Reconstruction:
         A Comparative Evaluation and Empirical Validation},
  author={{Anonymous}},
  year={2026},
  note={Manuscript under review}
}
```

## 7. License

Available for academic use. Please contact the authors for commercial licensing.
