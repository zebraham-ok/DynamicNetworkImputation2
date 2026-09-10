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
  `description`, `name`, `country`, `industry_1st`, `industry_2nd`.
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
```

Results land in `results/<model>/sampling_<timestamp>/<index>_<mode>_<scratch|frozen>/`,
summarised in `summary.yaml`. Add `--resume` to skip already-finished modes, or
`--num_rnn_layers 2` to reproduce the two-BiGRU-layer variant.

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
```

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
