# Technical Guide — 技术架构与模块接口手册

适用范围：本仓库当前改造版（`opensource-revise`，分支 `revise`）。
本文件只描述**结构、接口与参数归属**，不重复论文口径的数字（那些在 `README.md` 与论文正文）。

> ## 维护约定（强制）
> 1. 任何代码 / 配置改动完成后，**必须在同一次改动里同步更新本文件**对应小节。
> 2. 凡是新增或改名了 **配置键 / 函数签名 / 返回值结构 / checkpoint 字段 / TensorBoard 标签**，
>    本文件是唯一登记处；没有登记就等于后续无法审计。
> 3. 修改注册表见 §9；发现本文件与代码不一致时，以代码为准并立刻回填。
> 4. 本文件对外可读：不写内网地址、服务器信息、厂商名与具体运行时间戳。

---

## 0. 一页速览

### 0.1 入口一览

| 入口 | 用途 | 读取的配置 | 训练器 | 主要产物 |
|---|---|---|---|---|
| `SampleSetting/run_sampling.py` | **论文主协议**：4 种负采样模式 × 共享 backbone，两阶段；`--repeats N` 时在同一 run 根内重复 N 次（多种子，§5.8），**默认开启 backbone 供体轮换**（每 repeat 换一个模式当 donor，§5.8.1） | `SampleSetting/sample_setting.yaml` + `Training/common_config.yaml` ← `Models/configs/{model}.yaml` | `DynamicGraphTrainer` | `results/<subdir>/sampling_{MMDD-HHMM}/`（`--repeats>1` 时每种子一个 `seed<SEED>/`） |
| `Training/train_{gatgru,gatgru_1dire,bigru,egcn,tna}.py` | 单配置基线训练（不经 SampleSetting，但共用同一选模/调度协议） | 同上 | `DynamicGraphTrainer` | `results/<model>/...` |
| `Training/train_dualseal.py` | Temp-SEAL 专用训练 | 同上 | `SEALTrainer`（自带 lr 调度） | SEAL run 目录 |
| `Bootstraps/bootstrap_v2_train.py` | Bootstrap v2（B 次重采样训练） | `Bootstraps/bootstrap_v2_config.yaml` | `DynamicGraphTrainer` / 复用入口 | `results/bootstrap_*` |
| `Prediction/find_threshold*.py` | 阈值扫描（F1-max 等） | 模型 config（`load_config`） | — | 阈值与曲线 |
| `Prediction/get_imputation_fast.py` | 全图连边打分（内部数据补全） | 模型 config + `Prediction/imputation_common.yaml` | — | 边分数 / 补全网络 |
| `Analysis/visualize_results.py` | 读 eventfile → Table 3 / 汇总表；`--multi-run confidence` 画均值±std 带、`--seed-ci` 出跨 run mean ± 95% CI | — | — | `summary_table.*`、`best_metrics_table.csv`、`seed_ci_{per_run,summary}.csv` |
| `Analysis/seed_ci_summary.py` | 读每个 repeat 的 `summary.yaml`（不依赖 TensorBoard）→ 多种子 mean ± 95% CI 表/图 | — | — | `seed_metrics.csv`、`seed_summary.{csv,md}`、`seed_ci_bars.png` |
| `Analysis/PrROC_all_models.py` | 扫 `results/<model>/`，取每个模型 test AUC 最高的负采样配置，画多模型 ROC（+EdgeBank 基线） | `model_predictions_best_auc.npy` | — | `fig_roc_all_models.png`、`roc_curves_all_models.csv`、`roc_best_auc_summary.csv` |

### 0.2 端到端数据流

```
Neo4j (semi 供应链图)
   │  Data/neo4j_SPLC.py            # 连接与查询封装
   ▼
Data/graph_dataset.py
   CompanySupplyDataset             # 正样本 / 负样本 / 年份分布 / factset 边
   assemble_node_features           # X = [embedding | one-hot(country) | one-hot(industry_2nd) | one-hot(category_3rd)]
   build_static_graph               # 静态图对象（x, edge_index, edge_time, num_nodes）
   ▼
Data/company_dataset.py
   create_bootstrap_datasets        # 划分 + 负样本池 + 黑名单 + val/test 固定负样本
   create_dataloaders   ────────►   (static_data, train_loader, val_loader, test_loader, full_dataset)
   ▼
SampleSetting/run_sampling.py       # Phase 1：donor 模式全参训练 → backbone.pth
   │                                # Phase 2：冻结 static_encoder/feature_extractor，只训 temporal_encoder + edge_predictor
   ▼
Training/trainer_common.py :: DynamicGraphTrainer
   train_epoch → evaluate(val) → evaluate(test) → FactSet 统计
   → 选模分值（§6.3）→ 存 best_model.pth + 早停 → LR 调度 step
   ▼
results/<subdir>/sampling_{MMDD-HHMM}/
   {idx}_{mode}_{scratch|frozen}/
       best_model.pth             # 复合/统一准则选出的 ckpt（含完整指标字典）
       best_auc_model.pth         # 选模 split AUC 最高的 ckpt（旁路产物）
       model_predictions_best*.npy
       *_frozen/ 或 *backbone/    # eventfile（TensorBoard）
   backbone.pth                   # Phase 1 导出的共享 backbone
   summary.yaml / config.yaml / sample_setting.yaml
   seed<SEED>/                     # 仅 --repeats > 1：每个 repeat 各有一份上面的结构（含自己的 backbone.pth）
                                   #   轮换开启时 _scratch 落在哪个模式由该 repeat 的 mode_order 决定（§5.8.1）
   ▼
Analysis/visualize_results.py  →  Table 3 / 附录图（需先把 eventfile 放到 <model>/<MMDD-HHMM>-<flag>/）
```

---

## 1. 目录与职责

| 路径 | 职责 | 禁止 |
|---|---|---|
| `Data/` | 图读取、数据集、负采样、特征装配、数据加载器 | 不写训练逻辑；不 import `Training/` |
| `Models/` | 6 个 backbone + 2 个融合模型（§1.2）+ 共享组件；每个模型一个 `configs/*.yaml` | 不直接读 Neo4j；不自行实现特征投影 |
| `Training/` | 共享训练器、SEAL 专用训练器、6 个单配置训练脚本、配置合并与共享配置 | 不改数据语义 |
| `SampleSetting/` | 论文主协议入口（4 模式 × 两阶段）与模式定义 | 模式定义只此一处 |
| `Bootstraps/` | Bootstrap v2 训练与稳定性分析 | — |
| `Prediction/` | 阈值扫描、全图补全 | — |
| `Analysis/` | 表格/图/曲线脚本（只读产物） | 不参与训练 |
| `results/` | 训练产物（`results/` 已被 `.gitignore` 忽略） | 不手工改数字 |

### 1.1 六个 backbone

| config 名 | 模块 | 类 | 备注 |
|---|---|---|---|
| `gatgru_vec` | `Models.gatgru_vectorized` | `BiGRUImputationWithGATVectorized` | 论文主模型（GAT-GRU，1 层 BiGRU） |
| `gatgru_1dire` | `Models.gatgru_1dire` | `UniGRUImputationWithGATVectorized` | 单向 GRU（GAT-GRU†） |
| `bigru_vec` | `Models.nodegru_vectorized` | `BiGRUImputationVectorized` | Node-GRU |
| `egcn` | `Models.egcn` | `EGCN_PyG_Vectorized` | EvolveGCN-H |
| `tna_vec` | `Models.grutna_vectorized` | `BiTNAImputationVectorized` | TNA |
| `seal` | `Models.tempseal` | `SEALWithTemporalWeighting` | Temp-SEAL，**forward 签名不同**（§5.2），走 `SEALTrainer` |

`num_rnn_layers: 2`（`--num_rnn_layers 2` 或 `sample_setting.yaml model_overrides`）产生 GAT-GRU\*（双层），与主模型分开报告。

### 1.2 融合模型（单分支，2026-09-12 新增）

把 EvolveGCN-H 的**算子演化机制**并入 GAT-GRU**内部**，而不是在 pair head 层做晚期融合。
设计依据 `results/FUSION_DESIGN_OPTIONS_0912.md`（§7 = 方案 3，§8 = 方案 4）。

| config 名 | 模块 | 类 | 机制 | 参数量 |
|---|---|---|---|---|
| `fusion_temporal_injection` | `Models.fusion_temporal_injection` | `TemporalInjectionGATGRU` | 方案 3（TI）：`W_t` 直接作用于注意力输出 `u_t = act(s_t W_t + A_t (s_t W_t))`，再进 BiGRU | 748,801 |
| `fusion_film` | `Models.fusion_film` | `FiLMGATGRU` | 方案 4（FiLM）：只由 `W_t` 生成一对调制参数 `s'_t = (1+γ_t)⊙s_t + β_t` | 847,617 |

共同约定（改动前请连读 §4.4）：

- **复用不复制**：`MatGRUCell_PyG` 直接 import 自 `Models/egcn.py`，`precompute_adj_matrices` 取自 `Models/common_vectorized.py`。EvolveGCN-H 的实现若变化会自动传导到这两个模型。
- **冻结前缀对齐**：`feature_extractor` / `static_encoder` 进 `backbone.pth`；**算子演化（MatGRU）+ GRU 全部塞进 `temporal_encoder`**，因此 Phase 2 会把方案 3/4 的全部时序部件一起重初始化。若把 MatGRU 放到别处，frozen 模式的数字就与 GAT-GRU 不可比。
- **forward 签名**：与 §4.2 的"常规"签名相同（`forward(link_indices, current_times)`），trainer 无需改动。
- **FiLM 恒等初值**：`summarizer.mlp` 末层零初始化 ⇒ 初值 `γ=β=0`、`s' ≡ s`，起点严格等价于 GAT-GRU（自检 `_tmp_dump/test_fusion_models.py` 实测 `max|s'−s| = 0`）。
- **`summary_mode`**（仅 `fusion_film`）：`weights`（默认，`g_t = mean_dim0(W_t)`，走 MatGRU，+0.163 M）或 `attention`（`g_t = mean_N(s_t)`，几乎零成本，靠逐年 `A_t` 变化获得时间性）。**不提供 `mean_N(z)` 摘要**——共享编码器逐年使用同一个 `z`，该摘要对 `t` 恒定，会使"逐时间步调制"失效。
- **宽度约束**：`fusion_temporal_injection` 要求 `static_hidden_dim == fc_embed_dim`（`W_t` 由 `mean_N(z)` 驱动却作用于 GAT 输出），构造时显式 `ValueError`；`fusion_film` 无此约束。
- **运行**：`python SampleSetting/run_sampling.py -c fusion_temporal_injection -s fusion_temporal_injection -d cuda:0`；两个 yaml 的 `output.subdir` 只覆盖自身，**换 config 必须显式 `-s`**（§8）。

---

## 2. 配置系统

### 2.1 合并规则（唯一入口）

```python
from Training.config_loader import load_config
cfg = load_config('gatgru_vec')   # = deep_merge(common_config.yaml, Models/configs/gatgru_vec.yaml)
```

- **按键深合并，同名键模型文件赢**（`utils.deep_merge`）。
- 因此"同一参数听谁的"由**键路径**决定，不是文件优先级：不同键路径即使同名（如 `trainer.patience` 与 `trainer.kwargs.patience`）会**共存**且被不同消费方读取。
- 所有读取模型配置的脚本都走 `load_config`（旧版 4 处直接 `yaml.safe_load(Models/configs/*.yaml)` 已统一）；
  `Prediction/get_imputation_fast.load_imputation_config` 是等价的 inline 三段合并（common → model → `imputation_common.yaml`）。

### 2.2 键归属

| 键 | 定义处 | 消费方 | 说明 |
|---|---|---|---|
| `dataset.*` | common（模型可覆盖） | `create_dataloaders` | batch size、负采样开关、特征开关 |
| `dataset.use_attr_onehot` / `onehot_attrs` | common | `dataset_feature_kwargs` | 决定 X 的宽度 |
| `model.kwargs.*` | common（共享编码器）+ 模型 yaml | `resolve_auto_kwargs` → 模型构造 | 共享编码器参数**只有 common 一处** |
| `trainer.num_epochs` / `trainer.patience` | common（模型可覆盖） | `DynamicGraphTrainer.train()` | 早停预算 |
| `trainer.kwargs.margin_lambda` | common | `DynamicGraphTrainer.__init__` | BCE + Margin 权重 |
| `trainer.selection.*` | common | `DynamicGraphTrainer` | **选模/早停准则**（§5.3） |
| `trainer.lr_schedule.*` | common | `DynamicGraphTrainer` | 学习率与调度（§5.4） |
| `trainer.max_factset_edges` | common | `compute_factset_quantile` | FactSet 抽样上限 |
| `output.*` | `sample_setting.yaml` | `run_sampling.py` | 输出目录（注意 `subdir` 写死，§8） |
| `defaults.repeats` / `defaults.seed` | `sample_setting.yaml` | `run_sampling.py` | 重复运行次数与首种子（`--repeats/--seed` 覆盖）；`repeats>1` 启用 `seed<SEED>/` 布局（§5.8） |
| `defaults.rotate_donor` | `sample_setting.yaml` | `run_sampling.py` | 是否每 repeat 轮换 backbone 供体（`--rotate_donor/--no_rotate_donor` 覆盖）；**只对 `repeats>1` 有效**（§5.8.1） |

### 2.3 生效配置落盘（可审计）

`run_sampling.py` 在训练开始前写 `<run>/config.yaml`（全量合并后的 `merged_config` + `run` 元信息 + 4 模式开关摘要）；
resume 时另写 `config_resume_{timestamp}.yaml`，保留首次的 `config.yaml`；`--dry_run` **完全不落盘**。

---

## 3. 数据层接口

### 3.1 `create_dataloaders`（唯一取数入口）

```python
# Data/company_dataset.py
create_dataloaders(
    negative_ratio=1, embedding_name="embedding", batch_size=32, train_ratio=0.8,
    toy_mode=False, random_state=None, min_degree=2, source_filter='semi',
    other_possible_fill=0.0, filter_factset_neg=False, intra_industry_neg=True,
    use_pred_neg=True, neg_dir=None, use_attr_onehot=False, onehot_attrs=None,
) -> (static_data, train_loader, val_loader, test_loader, full_dataset)
```

**返回值是 5 元组**，所有调用方必须按此解包。`static_data` 同时作为模型输入（`dynamic_data`）与 trainer 的 `static_data` 传入（Temp-SEAL 需要）。

特征开关统一由 `dataset_feature_kwargs(ds_cfg)` 收集，保证训练与推理的 X 布局不会漂移。

### 3.2 数据集组成

| split | 组成 |
|---|---|
| train | 正样本（按年重采样后的 train_pool）+ **等量动态负样本**（1:1）+ `use_pred_neg` 时的 CSV 负样本（train 份额） |
| val / test | 正样本 + **等量固定静态负样本（1:1）** + `use_pred_neg` 时 CSV 负样本（各 0.1 份额）；**不含动态负样本**（`negative_ratio=0`） |

- val/test 的组成有 **自检断言**（`create_bootstrap_datasets` 内），组成被改动会直接抛错。
- `negative_ratio` 参数对 val/test **无效**（仅train的有效性由 `BootstrapIterationDataset` 内部处理）。
- 黑名单：val/test 的正负样本（含 CSV 份额）一律不参与训练。
- `BootstrapIterationDataset.__getitem__`：`idx < total_pos` → 正样本；否则 → 动态负样本；再往后 → CSV 预定义负样本。

### 3.3 负采样开关与模式命名（唯一来源）

模式名 = 三个开关的缩写，顺序固定（`SampleSetting/sample_setting.yaml`）：

| 开关 | 含义 |
|---|---|
| `filter_factset_neg` | 负采样时是否避开 FactSet 边 |
| `intra_industry_neg` | 是否使用行业内负采样 |
| `use_pred_neg` | 是否注入 CSV 预定义负样本 |

`ftt`（后一项为 t）是唯一 `use_pred_neg: true` 的模式。**任何文档/图注/代码里的开关值都必须从 YAML 读，不得硬编码。**

### 3.4 图对象与样本结构

- 静态图对象：`x`（N × d）、`edge_index`（2 × E）、`edge_time`、`num_nodes`。
- `x` 的构成（`dataset.use_attr_onehot=true` 时）：`[embedding(128) | one-hot(country) | one-hot(industry_2nd) | one-hot(category_3rd)]`，唯一装配点 `graph_dataset.assemble_node_features` → `CompanySupplyDataset._build_attr_onehot`。
  - **属性缺失 → 该块整行为 0**（2026-09-12 起；词表宽度 = 非空取值个数，每块不再保留 `<NA>` 类，块内**不**保证恰有一列激活）。因此块宽度 = 取值数，X 宽度随之改变 ⇒ 旧 ckpt / `backbone.pth` 失效。
- 样本：`(link_indices: LongTensor[B, 2], current_times: FloatTensor[B], label: FloatTensor[B])`。
- 时间编码：`time_var = (year - 起始年) / 年数`，由 trainer 侧按 `current_times` 传入模型。

---

## 4. 模型层接口

### 4.1 共享节点特征编码器（所有 backbone 共用）

```python
# Models/common.py
build_feature_extractor(input_dim, enabled=True, output_dim=128, hidden_dim=None,
                        dropout=0.3, num_layers=3) -> NodeFeatureExtractor | None
apply_node_feature_extractor(extractor, x) -> Tensor
```

- 结构：`[Linear(in, hidden) - ReLU - Dropout] × (n-1)` + `Linear(hidden, out)`；`hidden_dim=None → output_dim * 2`。
- 参数落点：`Training/common_config.yaml → model.kwargs`（**唯一可改处**），模型 yaml 不得重复声明。
- 输入 `input_dim = dynamic_data.x.size(1)`（由 `resolve_auto_kwargs` 的 `auto` 解析，见 §4.3）。

### 4.2 forward 签名（两种，trainer 自动识别）

| 模型 | 签名 |
|---|---|
| 5 个常规 backbone | `forward(link_indices, current_times) -> pred[B]` |
| Temp-SEAL | `forward(data, link_indices, current_times) -> pred[B]` |

`DynamicGraphTrainer._forward_takes_static_graph()` 用 `inspect.signature` 判定，`_predict()` 是**唯一推理入口**（训练/验证/测试/FactSet/存 npy 全走它）。

### 4.3 auto 参数解析

`utils.resolve_auto_kwargs(kwargs, context)`：值等于字符串 `'auto'` 时从 `context` 取同名键。`run_sampling.py` 提供的 context：

```python
{'num_features': x.size(1), 'num_nodes': N, 'time_steps': [...], 'hidden_dims': x.size(1), 'device': ...}
```

### 4.4 两阶段协议与冻结

| 阶段 | 训练参数 | 冻结 |
|---|---|---|
| Phase 1（donor scratch） | 全部 | — |
| Phase 2（其余 3 模式） | `temporal_encoder.*` + `edge_predictor.*`（SEAL：`dynamic_gnn.*` + `link_predictor.*`） | `static_encoder.*` / `static_gnn.*` / `feature_extractor.*` |

- 冻结前缀定义在 `Data/company_dataset.load_pretrained_backbone`；`reinit_trainable_parts()` 重置可训部分。
- `backbone.pth` 只存 `static_encoder.* / static_gnn.* / feature_extractor.*` 权重（`save_backbone`），可解释为"共享补全骨干"。
- **改 X 的宽度或共享编码器结构 ⇒ 旧 ckpt / backbone.pth 全部不兼容**，必须重跑 Phase 1。

---

## 5. 训练器接口（`Training/trainer_common.py :: DynamicGraphTrainer`）

### 5.1 构造参数

```python
DynamicGraphTrainer(
    model, train_loader, test_loader, val_loader=None,
    device=..., log_dir=None, use_tensorboard=True,
    margin_lambda=0.1, margin=1.0,
    factset_edges=None, node_mapping=None, reverse_node_mapping=None,
    static_data=None,
)
```

- `val_loader=None` ⇒ `selection_split='test'`（旧协议回退）。
- `static_data` 仅供 Temp-SEAL 使用；签名不匹配但未传静态图会立刻抛错。
- 只有当 `factset_edges` 与 `node_mapping` 同时存在时才会计算 FactSet 统计。

### 5.2 `train()`

```python
train(num_epochs, save_path='best_dynamic_graph_model.pth',
      patience=10, max_factset_edges=None) -> (best_epoch, best_score)
```

每个 epoch 固定顺序：

1. `train_epoch()`（BCE + `margin_lambda × MarginRankingLoss`）
2. `evaluate(val_loader)`（若无 val 则跳过）
3. `evaluate(test_loader)`（**只报告**）
4. `compute_factset_quantile()`（选模 split + test 各一份，复用第 2/3 步的缓存分数）
5. 计算选模分值 → 存 ckpt / 早停判定 → LR 调度 `step`
6. 记录 `history` 与 TensorBoard

### 5.3 选模 / 早停准则（**统一准则 + EMA**）

单一准则同时决定 checkpoint 与早停，避免"两个排序打架 + 共用一个 patience 计数器"。

```
select_metric = AUC(selection_split)            # 默认；trainer.selection.metric 可改为 f1
ema_span      = trainer.selection.ema_span      # <=1 表示不平滑
smooth(x)     = EMA(x, span)                    # 首值直接取 raw，避免零偏置
score         = smooth(select_metric)                              if use_factset=false
              = 0.5·smooth(FactSet quantile) + 0.5·smooth(select_metric)   if use_factset=true
```

- `trainer.selection.use_factset: false` ⇒ **FSQ 不参与选模**，但**仍照常计算、记录、写入 ckpt 与 TensorBoard**（Table 3 的该列不受影响）。
- 最高分 ⇒ 覆盖写 `best_model.pth` 并把 `patience_counter` 清零；否则 `patience_counter += 1`，达到 `trainer.selection.patience`（为 `null` 时回落 `trainer.patience`）即停。
- `best_auc_model.pth` 是**旁路产物**：只按选模 split 的**原始 AUC** 取最高，不参与早停计数（供曲线/阈值脚本使用）。
- 历史字段 `history[i]` 至少含：`epoch, selection_split, train_bce, train_margin, train_f1, train_auc, val_*, test_*, factset_quantile, wasserstein_diff, factset_quantile_test, wasserstein_diff_test, epoch_seconds, score_raw, score, lr`。

### 5.4 学习率调度

| 键 | 值域 | 说明 |
|---|---|---|
| `enabled` | bool | 是否按 `name` 衰减；`false` ⇒ 恒定 `base_lr` |
| `name` | `cosine` / `step` / `plateau` / `none` | 见下 |
| `base_lr` | float | 初始学习率（`enabled: false` 时也生效） |
| `min_lr` | float | 下限 |
| `warmup_epochs` | int | 前 N 个 epoch 线性升到 `base_lr` |
| `total_epochs` | int | cosine 周期 |
| `milestones` / `gamma` | list / float | step 衰减点与倍率 |
| `plateau_factor` / `plateau_patience` | float / int | `ReduceLROnPlateau`（单位 = **epoch**，metric = 平滑后的选模分） |

- 调度在 **epoch 开始时**应用（`step`/`cosine`/`warmup` 为确定性公式）；`plateau` 在 epoch 评估后 `step(score)`，此时 `_apply_epoch_lr` 自动让位，不会把调度器的 lr 覆盖回去。
- `cosine` 超出 `total_epochs` 后**钳在 `min_lr`**（不会回升）；`base_lr: 0` 也是合法值（冻结权重，离线自测用），不会被默认值顶替。
- **早停 budget 必须大于调度 patience**，否则 lr 永远等不到衰减就被早停掐掉（`patience: 15` vs `plateau_patience: 3`）。启动时会打印告警。
- 学习率每 epoch 写入 `Train/Epoch_LR`（epoch 级），并进 `history` 的 `lr` 字段；`plateau` 的每次真实下降还会进 `trainer.lr_events` 与 `summary.yaml`。

### 5.5 checkpoint 字段（`best_model.pth`）

| 类别 | 字段 |
|---|---|
| 结构 | `model_state_dict`, `optimizer_state_dict`, `epoch`, `selection_split` |
| 训练量 | `train_bce`, `train_margin`, `epoch_seconds` |
| 选模 split | `val_loss`, `val_f1`, `val_auc` |
| test（只报告） | `test_loss`, `test_f1`, `test_auc` |
| 准则登记 | `selection_criterion`（如 `EMA5(AUC)`）、`selection_score`（本次选模分）、`selection_score_raw`（未平滑）、`selection_metric`、`selection_use_factset`、`selection_ema_span`、`lr` |
| FactSet | `factset_scope`, `factset_quantile`, `factset_wasserstein_{pos,neg,diff}` 及其 `_test` 版本 |

下游硬依赖：`Prediction/find_threshold*.py` 读 `val_auc/test_auc`；`run_sampling` 读 `selection_score` 与全部指标。**新增字段可，改名/删除不可。**

### 5.6 TensorBoard 标签约定

| 标签 | 粒度 | 含义 |
|---|---|---|
| `Train/Batch_{BCE,Margin,F1,AUC}` | 每 10 batch | 训练侧 batch 曲线 |
| `Train/Epoch_{BCE,Margin,F1,AUC}` / `Train/Epoch_LR` | epoch | 训练侧 epoch 汇总 |
| `Val/Epoch_{Loss,F1,AUC}`, `Test/Epoch_{Loss,F1,AUC}` | epoch | 两个评测 split（不用作选模） |
| `Monitor/*` | epoch | **选模 split** 的监控量：`Factset_Quantile`、`Wasserstein_*`、`Selection_Score`、`Selection_Score_EMA` |
| `MonitorTest/*` | epoch | test split 的同名监控量（与 `Monitor/*` 永不共用曲线） |
| `Val|Test/Batch_{Loss,F1,AUC}` | 每 10 batch | 仅供 SEAL 这类长 epoch 曲线 |
| `Time/Epoch_Duration` | epoch | 每 epoch 秒数 |
| `Score_Distribution/*` | epoch | 分数直方图（FactSet / 各 split 正负） |

### 5.7 `SEALTrainer`（`Training/train_dualseal.py`）差异

- 自带 `ReduceLROnPlateau(mode='max', factor=lr_factor, patience=lr_patience, min_lr)`，在 **step 级周期评估**处 `step(选模 split F1)` ⇒ `lr_patience` 单位是**评估次数**，不是 epoch。
- `SEALTrainer` 读 `trainer.kwargs.*`（`seal.yaml` 的 `kwargs.patience` 只对它生效），`DynamicGraphTrainer` 读顶层 `trainer.*`；**`seal.yaml` 的 `trainer.kwargs.patience: 15` 对 `run_sampling.py` 无影响**。
- 早停计数器目前未接上（占位字段）；调 `-c seal` 走 `run_sampling.py` 时 epoch 预算由 `seal.yaml` 的 `trainer.num_epochs` 决定。

### 5.8 多种子重复运行（`run_sampling.py --repeats N`）

一句话：**`--repeats N` 在同一个 run 根下把整个两阶段流程重复 N 次**，每次换种子、重新初始化模型，并把 backbone 等产物放进各自的 `seed<SEED>/`，所以不同种子的权重永不互相覆盖。默认 `N = 1`，布局与历史完全一致。

- 种子计划：`seeds = [seed, seed+1, ..., seed+N-1]`（`resolve_seed_plan`）；`seed` 来自 `--seed` 或 `defaults.seed`（默认 42）。
- 目录：`seed_run_dir()` → `repeats == 1` 时**就是 run 根**（历史布局，既有分析与回传脚本不受影响）；`repeats > 1` 时为 `<run 根>/seed<SEED>/`。
- 每个 repeat 自含：`backbone.pth`、`{idx}_{mode}_{scratch|frozen}/`（ckpt + eventfile）、`summary.yaml`（`_meta` 含 `seed`、`output_dir`、`selection_criterion`）。
- run 根额外写 `summary.yaml` = `{'_meta': {..., 'seed_dirs'}, 'aggregate': {...}}`，其中 `aggregate = aggregate_seed_results()` = `{mode_key: {metric: {mean, std, ci95, n, values}}}`（CI 为 t 分位数；单种子时 `std/ci95 = None`）。
- 种子只影响随机初始化与负采样顺序，**不改变选模/早停协议（§5.3）与 LR 调度（§5.4）**；`run_single_training(..., seed=)` 一个形参贯穿 `torch.manual_seed` / `np.random.seed` / 模型构造 / `create_dataloaders`。
- resume：`discover_seed_dirs()` 扫到 `seed*/` 就只重跑这些种子；历史单次布局则从自己的 `summary.yaml` 读回 `_meta.seed`。resume 时 `--repeats/--seed` 被忽略并打印提示。

#### 5.8.1 backbone 供体轮换（`--rotate_donor`，默认开）

一句话：**每 repeat 把模式进入顺序整体左移一位**（最先进入的那一个挪到最后），于是"谁来提供 backbone"也跟着换，避免某个模式永远只能吃别人训出来的骨干。

- 顺序：`resolve_mode_order(modes, donor_idx, repeat_idx, rotate)`。不轮换时 = `[donor_idx, ...其余按 YAML 顺序]`；轮换时在 `[donor_idx, ...]` 这个锚点上再左移 `repeat_idx % len(modes)` 位 ⇒ 4 个模式时 4 次 repeat 正好让每个模式当一次 donor：
  ```
  repeat 1 (seed42): [0]ftt scratch -> [1]ftf -> [2]fff -> [3]tff
  repeat 2 (seed43): [1]ftf scratch -> [2]fff -> [3]tff -> [0]ftt
  repeat 3 (seed44): [2]fff scratch -> ...
  repeat 4 (seed45): [3]tff scratch -> ...
  ```
- 生效范围：`repeats == 1` 时**必然是 `[donor_idx, ...]`**，所以默认单次运行的历史行为完全不变；`--backbone_donor k` 在轮换时表示"repeat 1 的 donor"，其余 repeat 依次顺移（轮换可关：`--no_rotate_donor` 或 `defaults.rotate_donor: false`）。
- 每个 repeat 独立：`_run_one_repeat(..., mode_order=)` 用本 repeat 的顺序重算 `donor_idx/donor_mode/frozen_modes`（含 `_scan_completed_modes` 的 resume 扫描），所以 `seed*/` 里 `_scratch` 落在哪个模式随 repeat 变化，`{idx}_{name}_scratch|frozen` 后缀仍如实反映该 run 的角色。
- 落盘：`_meta.rotate_donor`（run 级与 per-repeat）、`_meta.mode_order = {seed: [idx...]}`、per-repeat `_meta.donor_mode` / `_meta.frozen_modes`；`config.yaml` 的 `run` 段同样记录，**resume 时优先用 `_read_recorded_mode_orders()` 读回的顺序**（CLI 显式覆盖会告警），否则会用今天的默认值算出错误目录。
- 聚合口径（关键）：轮换后同一个模式在不同 repeat 里是不同角色，**按 `{idx}_{name}_{role}` 精确键聚合会退化成 n=1**，所以 run 根 `summary.yaml` 同时写两份：
  - `aggregate`：精确键（`aggregate_seed_results`），看单个处理的重复性；
  - `by_mode`：**按模式合并角色**（`aggregate_by_mode`）= `{mode: {seeds: {scratch:[...], frozen:[...]}, metrics, by_role}}`，每个模式都是"1 次 scratch + N-1 次 frozen"，这就是供体轮换后的**平衡均值**，也是四种模式横向比较该用的数。
  - 代价：`by_mode` 把两种训练制度（全参 vs 冻骨干）混在一个均值里，故同时给 `by_role`；论文口径要写明用的是哪一种（`_format_by_mode_report` 同时打印 pooled 与 frozen-only 两列）。
  - `Analysis/seed_ci_summary.py` 自动识别轮换（`_meta.rotate_donor`，或同一 run 下出现多个不同 `_scratch` 键）→ **按模式而非按精确键分组**，并把每个 run 的角色留在 `role` 列。
- 消费方：`Analysis/seed_ci_summary.py`（读 `summary.yaml`）与 `Analysis/visualize_results.py --seed-ci`（读 eventfile），见 §6。


---

## 6. 产物 → 报告链

```
results/<model>/<MMDD-HHMM>-<flag>/events.out.tfevents.*      ← eventfile 必须直接落在 run 目录（EventAccumulator 不递归）
        │
        ├─ Analysis/visualize_results.py         → summary_table.png / best_metrics_table.csv(Table 3 七列) / per_model_*.png
        ├─ Analysis/plot_train_4metrics.py       → 附录 A 训练曲线（SEAL 必须用 */Batch_* 镜像）
        ├─ Analysis/PrROC2_fff.py                → ROC/PR 图（依赖 Analysis/edgebank_baseline_results.csv）
        ├─ Analysis/PrROC_all_models.py          → 跨模型 ROC（各模型取 test AUC 最高的配置）
        └─ Analysis/edge_imput_overlap_plot.py   → 重叠度-阈值图
summary.yaml（run 级，repeats == 1）→ 指标表重建；_meta_<ts>/summary.yaml 为回传归档布局
```

多种子（`--repeats > 1`）时的第二/第三条链：

```
sampling_*/summary.yaml（run 根）          → {_meta:{seed_dirs, donor_plan}, aggregate:{mode_key:{metric:{...}}},
                                              by_mode:{mode:{seeds, metrics, by_role}}}
sampling_*/seed<SEED>/summary.yaml         → Analysis/seed_ci_summary.py  → seed_summary.{csv,md} / seed_ci_bars.png
                                              （轮换时按 mode 分组、保留 role 列）
sampling_*/seed<SEED>/*_frozen/events.*    → Analysis/visualize_results.py --seed-ci
                                              （run 目录名需带 -s<seed> 后缀才写回 seed 列，见 §5.8）
```

- Table 3 的 7 列可从 `summary.yaml` + eventfile 重建（含 `avg_epoch_seconds`）。
- **多 run 聚合口径**：`visualize_results.py` 的 `_filter_multi_run(mode)`：`longest` 每个 (model, flag) 只留 epoch 最多的 run；`confidence` 全留（同一 run 名 + 不同种子 => mean ± std 带）。`--seed-ci` 再按"每个 run 取自己的 best epoch，再跨 run 求 mean ± 95% CI"出表（比一次性取 argmax 更保守，后者是最优种子的上偏值）。
- `visualize_results.py` 的 `best_criterion` 取 `Val/Epoch_AUC`（缺失回退 `Test/Epoch_AUC`）。
- **重跑口径变化（特征、选模 split、负样本实现）会让新旧数字不可比**，必须整表替换而非混用。

---

## 7. 参数索引表（改之前先查这里）

| 参数 | 位置 | 消费方 | 影响 |
|---|---|---|---|
| `dataset.batch_size` | 模型 yaml | DataLoader | 显存 / step 数 / 曲线粒度 |
| `dataset.use_attr_onehot`, `onehot_attrs` | common | `dataset_feature_kwargs` | **X 宽度 ⇒ 旧 ckpt 不兼容** |
| `dataset.filter_factset_neg`, `intra_industry_neg`, `use_pred_neg` | `sample_setting.yaml`（模式定义） | `create_dataloaders` | 负样本分布（4 模式） |
| `dataset.min_degree`, `source_filter` | common | 正样本池 | 样本量 |
| `model.kwargs.use_fc_embedding/fc_embed_dim/fc_hidden_dim/fc_num_layers` | common | `build_feature_extractor` | 共享编码器 ⇒ 旧 ckpt 不兼容 |
| `model.kwargs.*`（模型自身） | 模型 yaml | 模型构造 | 容量/层数 |
| `trainer.num_epochs` | common（模型可覆盖） | trainer | 训练预算 |
| `trainer.patience` | common | trainer（早停回退值） | 早停预算 |
| `trainer.selection.*` | common | trainer | **选模/早停准则** |
| `trainer.lr_schedule.*` | common | trainer | 学习率轨迹 |
| `trainer.max_factset_edges` | common | FactSet 统计 | 监控指标方差 |
| `trainer.kwargs.margin_lambda` | common | trainer | BCE 与排序损失权衡 |
| `save_test_predictions` | common | 训练后 | 是否落 npy |
| `pretrained.path` | common | 训练前 | 是否加载并冻结骨干 |
| `defaults.repeats`, `defaults.seed` / `--repeats`, `--seed` | `sample_setting.yaml` / CLI | `run_sampling.py` | 重复运行次数与种子计划；`>1` 时启用 `seed<SEED>/` 隔离布局（§5.8） |
| `defaults.rotate_donor` / `--rotate_donor`, `--no_rotate_donor` | `sample_setting.yaml` / CLI | `run_sampling.py` | 每 repeat 换一个 backbone 供体（`resolve_mode_order`）；`repeats==1` 无效，resume 时以 `config.yaml` 记录为准（§5.8.1） |
| `IMPUT_MULTI_RUN` / `--multi-run {longest,confidence}` | 环境变量 / CLI | `visualize_results.py`、`plot_train_4metrics.py` | 同 (model, flag) 多个 run 时的取舍；`confidence` = 画 mean ± std 带 |
| `--seed-ci` | CLI | `visualize_results.py` | 跨 run best-epoch 的 mean ± 95% CI 表 + CSV（隐含 `--multi-run confidence`） |
| `--results-dir`, `--out-dir`, `--metrics`, `--fig-metrics`, `--no-figure` | CLI | `Analysis/seed_ci_summary.py` | 多种子汇总的输入目录/指标/图 |

---

## 8. 已知陷阱

1. **`sample_setting.yaml → output.subdir` 写死 `gatgru`**：换 backbone 必须加 `--subdir <name>`，否则写进 GAT-GRU 的目录。
2. **同键模型赢 / 不同键共存**：`seal.yaml` 的 `trainer.kwargs.patience` 与顶层 `trainer.patience` 是两条键路径，分别被 SEALTrainer 与共享 trainer 读取。
3. **`run_sampling.py` 恒用 `DynamicGraphTrainer`**：不读 config 的 `trainer.module/class`，所以"配置里写的 trainer 类型"不会生效。
4. **协议入口必须成组检查**：`run_sampling.py` 与 5 个 `train_*.py` 都会传 `trainer.selection.*` / `trainer.lr_schedule.*`（2026-09-12 起统一）。
   - 新增入口时如果漏传这两个键，会**静默回到旧准则**（0.5×FSQ + 0.5×AUC、恒定 lr），而日志里看不出区别。`DynamicGraphTrainer` 的默认值就是旧协议，这是有意的向后兼容设计。
   - `SEALTrainer`（`train_dualseal.py`）不读这两个键，自成一套（`trainer.kwargs.lr_factor/lr_patience`，单位是评估次数）。
5. **FactSet 抽样用全局随机源**（`random.sample`）⇒ 单次运行的该列不可与其它运行逐位比较。
6. **`seal.yaml` 的 `num_epochs: 2`** 会让 Temp-SEAL 严重欠训练，其指标不可与其它模型并列。
7. **改 X 宽度 / 编码器结构 / 冻结前缀** ⇒ 旧 `backbone.pth` 与 ckpt 全部失效。
8. **eventfile 目录层级**：`Analysis/visualize_results.py` 期望 `results/<model>/<MMDD-HHMM>-<flag>/` 且事件文件直接在该目录下，多一层子目录会读到空标签。
9. **Windows 下不启用 `num_workers`**（`create_dataloaders` 内部按平台判断）。
10. **`--dry_run` 不落盘**：审计一次配置时不要用 dry_run，否则没有 `config.yaml`。
11. **CSV 预定义负样本可用率低**（端点过滤 + 同年已是正边会大量丢弃）⇒ `use_pred_neg` 的差异可能没有统计功效，解释时要先看实际条数（运行日志会打印）。
12. **`results/`、`*.pth/pt/pyg/zip`、`Data/stopped_neg_sample.csv` 被 `.gitignore` 忽略**：发布前必须检查真实负样本 CSV 是否被替换为空模板。
13. **`seed<SEED>/` 布局只在 `--repeats > 1` 出现**：`repeats == 1` 仍然写 run 根（历史布局），因此"结果在 run 根还是 `seed*/` 下"取决于启动参数；分析/回传脚本两种都要支持（`Analysis/seed_ci_summary.py` 已支持）。
14. **同一 (model, flag) 下的多个 run 会被 `--multi-run confidence` / `--seed-ci` 合并**：不要把不同协议或不同批次的 run 放进同一目录再聚合（09-12 前后、改 `selection`/`lr_schedule`、脱敏前后都不可混）。事后靠 `summary.yaml → _meta.selection_criterion` 分辨；`report_seed_ci()` 对"同组里既有带种子又有不带种子"的情况会告警。
15. **置信区间的 n**：CI 由 t 分位数给出，`n < 5` 只能看离散度、不宜在论文里引用区间；`run_sampling.py` 在 `repeats < 5` 时会打印提示。
16. **轮换后不能按 `{idx}_{name}_{role}` 精确键做跨种子均值**：角色在 repeat 之间搬家 ⇒ 每个键只剩 n=1（`aggregate` 仍会打印出来，容易误读为"重复性差"）。跨种子看模式效果请用 `by_mode` / `seed_ci_summary.py`（按模式合并角色）。同理，轮换 run 的 `seed*/` 目录不能直接按 flag 名重排进 `results/<model>/<MMDD-HHMM>-<flag>/` 做跨 flag 对比：**同一 repeat 内 4 个 flag 的角色不同**，跨 repeat 才平衡。
17. **轮换改变的是"谁能吃到谁的骨干"，不是协议**：轮换下每个模式的均值里混了 1 次 scratch + N-1 次 frozen（`by_role` 分开存）；若要报"全部模式都在冻骨干条件下"的纯对比，用 `by_role.frozen`；反之纯 scratch 对比每个模式只有 1 个点。写论文时必须说明用的是哪一种口径。

---

## 9. 修改注册表（每次改动后回填）

| 日期 | 改动 | 涉及文件 | 接口影响 | 回填小节 |
|---|---|---|---|---|
| 2026-09-12 | **选模协议 P0 改造**：FSQ 开关 + 统一准则 + EMA 平滑 + 独立早停计数 | `Training/trainer_common.py`, `Training/common_config.yaml` | trainer 构造新增 `selection_cfg`；`selection_score()` 成为唯一选模入口；旁路 `best_auc_model.pth` 不再触动 patience | §5.3 |
| 2026-09-12 | 学习率调度实装（step/cosine/plateau） | 同上 | trainer 构造新增 `lr_schedule_cfg`；`base_lr` 由 `trainer.lr_schedule.base_lr` 决定且 `enabled: false` 也生效 | §5.4 |
| 2026-09-12 | 5 个单配置入口统一协议 | `Training/train_{gatgru,gatgru_1dire,bigru,egcn,tna}.py` | 各新增 2 个构造参数；未传则静默回旧协议 | §0.1, §8 |
| 2026-09-12 | 运行自描述 | `SampleSetting/run_sampling.py` | `run_single_training` 返回值新增 9 个键（criterion / lr / lr_events / epochs_run / 两份 effective config）；`summary.yaml` 新增 `_meta.selection`、`_meta.lr_schedule`；resume 优先读 ckpt 的 `selection_score` | §5.5, §6, §3.1 |
| 2026-09-12 | **多种子重复运行**：`--repeats N` / `--seed`，每 repeat 独立 `seed<SEED>/`（含自己的 backbone.pth、ckpt、eventfile、summary.yaml），run 根写跨种子 aggregate | `SampleSetting/run_sampling.py`, `SampleSetting/sample_setting.yaml` | 新增 CLI `--repeats/--seed` 与 `defaults.repeats/seed`；`run_single_training` 新增 `seed` 形参；新增纯函数 `resolve_seed_plan/seed_run_dir/discover_seed_dirs/_seed_from_summary/_t95_quantile/aggregate_seed_results`；per-repeat `summary.yaml._meta` 新增 `seed/output_dir/selection_criterion/repeats/seeds` | §0.1, §0.2, §2.2, §5.8, §6, §7, §8 |
| 2026-09-12 | **多种子分析**：run 目录名可带 `-s<seed>`/`-seed<seed>` 后缀；多种子聚合开关；新增读 `summary.yaml` 的汇总脚本 | `Analysis/visualize_results.py`, `Analysis/plot_train_4metrics.py`, `Analysis/seed_ci_summary.py`（新增） | 新增 `parse_run_seed()`、`get_best_epoch_metrics_per_run()`、`aggregate_best_metrics()`、`report_seed_ci()`；`collect_all_data()` 记录新增 `seed` 列；`plot_summary_table(..., err_df=, err_label=)` 新增参数；`plot_train_4metrics.py` 新增 `--multi-run` | §5.8, §6, §7 |
| 2026-09-12 | **backbone 供体轮换**：每 repeat 把模式顺序左移一位，让每个模式当一次 donor（默认开，仅 `repeats>1` 生效；`--no_rotate_donor` 可关） | `SampleSetting/run_sampling.py`, `SampleSetting/sample_setting.yaml`, `Analysis/seed_ci_summary.py` | 新增纯函数 `resolve_mode_order/mode_label/_role_of/_read_recorded_mode_orders/_metric_stats/aggregate_by_mode/_format_by_mode_report/_format_rotation_matrix`；`_run_one_repeat` 新增 `ctx['mode_order']` 覆盖 donor/frozen；CLI 新增 `--rotate_donor/--no_rotate_donor` + `defaults.rotate_donor`；run 根 `summary.yaml` 新增 `by_mode` 与 `_meta.donor_plan`；`config.yaml` 的 `run` 段新增 `rotate_donor/mode_order`（resume 依据）；`seed_ci_summary.py` 新增 `_mode_label/_role_of/detect_rotated_runs`，`seed_metrics.csv` 新增 `role/rotate_donor/donor_mode` 列，轮换时按模式分组 | §0.1, §0.2, §2.2, §5.8.1, §6, §7, §8 |
| 2026-09-12 | 跨模型 ROC 图 | `Analysis/PrROC_all_models.py`（新增） | 每个模型取 test AUC 最高的 flag 画一条 ROC，附 Youden 点与 EdgeBank 基线；`--flag` 可强制统一配置 | §0.1, §6 |
| 2026-09-12 | **属性 one-hot 改为"缺失 = 全 0 行"**：不再为缺失保留 `<NA>` 类，块宽 = 非空取值个数（属性捷径的对照组） | `Data/graph_dataset.py`（`_build_attr_onehot` + 两处 docstring）、`Training/common_config.yaml`、`Bootstraps/bootstrap_v2_config.yaml`、`README.md` | 词表宽度变化 ⇒ X 宽度 **−3**（每块少 1 列），旧 ckpt 与 `backbone.pth` 全部失效，须重跑 Phase 1；`attr_onehot_vocabs/sizes` 语义不变但值变小；节点行不再保证 unit row sum；`describe_node_features()` 打印尾部新增说明 | §3.4 |
| 2026-09-12 | **融合单分支模型**：方案 3 时序注入（EvolveGCN-H 的 `W_t` 作用于 GAT 输出后再进 BiGRU）与方案 4 FiLM（只由 `W_t` 生成 `(γ_t, β_t)` 调制注意力输出），无晚期融合 | `Models/fusion_temporal_injection.py`（新增）、`Models/configs/fusion_temporal_injection.yaml`（新增）、`Models/fusion_film.py`（新增）、`Models/configs/fusion_film.yaml`（新增） | 纯新增，**不改任何既有接口**：沿用 §4.2 常规 forward 签名与 §4.4 冻结前缀；`MatGRUCell_PyG` 与 `precompute_adj_matrices` 为跨模块 import（未复制实现，`Models/egcn.py` 与 `Models/common_vectorized.py` 未改动）；`fusion_temporal_injection` 要求 `static_hidden_dim == fc_embed_dim`；`fusion_film` 新增 `summary_mode`/`mlp_hidden`/`film_unit_offset` 三个模型私有键（只写在自身 yaml，不进 `common_config.yaml`）；`FiLMGATGRU` 新增只读接口 `get_last_modulation()` / `get_last_modulated()` | §1.1, §1.2, §7 |
| — | （下次改动写在这里） | | | |

### 9.1 改动检查清单

- [ ] 新增/改名配置键 → 更新 §2.2 与 §7，并确认模型 yaml 的同类键不会静默覆盖。
- [ ] 改函数签名 / 返回值 → 更新 §3.1、§5.1、§5.2，全局检索调用点。
- [ ] 改 ckpt 字段 → 更新 §5.5，并检查 `Prediction/`、`Analysis/`、`Bootstraps/` 的读取点。
- [ ] 改 TB 标签 → 更新 §5.6，并检查 `Analysis/visualize_results.py`、`plot_train_4metrics.py` 的过滤规则。
- [ ] 改 X 宽度 / 编码器 / 冻结前缀 → 在 §8 标注"旧 ckpt 失效"，提醒重跑 Phase 1。
- [ ] 改选模或早停逻辑 → 在 §5.3 说明新旧数字是否可比。
- [ ] 本地自检三件套：`py_compile` 全部改动文件 → `run_sampling.py --dry_run`（确认启动打印里的 Selection / LR schedule 两行符合预期）→ 离线自测脚本（工作区临时目录 `_tmp_dump/test_selection_lr.py` 纯函数：EMA/调度公式/legacy 回退；`_tmp_dump/test_trainer_smoke.py` 玩具模型跑通 train()，核对 ckpt 字段、TB 标签与早停计数）。
- [ ] 新增 backbone / 融合模型 → 补齐 §1.1 或 §1.2 表格，并跑 `_tmp_dump/test_fusion_models.py`（合成图，零 Neo4j 依赖）：校验配置解析、`feature_extractor`/`static_encoder`/`temporal_encoder`/`edge_predictor` 命名契约、参数量、forward 形状与空 year 掩码、冻结前缀与重初始化语义、`DynamicGraphTrainer` 端到端 3 epoch 与 ckpt 字段。
- [ ] 改多种子相关（`--repeats`、**供体轮换 `--rotate_donor`**、run 名后缀、聚合口径）→ 另跑三件套：`_tmp_dump/test_seed_run_sampling.py`（编排侧，训练打桩：种子计划、`seed*/` 隔离、per-seed 与 run 根 `summary.yaml`、`resolve_mode_order` 轮换、角色分布、`by_mode` 平衡均值、resume 顺序回读）、`_tmp_dump/test_seed_analysis.py`（eventfile 侧：run 名解析、longest/confidence、per-run CI 算术、汇总表）、`_tmp_dump/test_seed_ci_summary.py`（`summary.yaml` 侧：两种布局、mean/std/ci95、n=1 降级、**轮换按模式分组**、md/png 产物）。
