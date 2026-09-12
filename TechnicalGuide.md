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
| `Training/train_dualseal.py` | Temp-SEAL 专用训练 | 同上 | `SEALTrainer`（自带 lr 调度；**epoch 上限与早停同样取自 common**，见 §5.7） | SEAL run 目录 |
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
       model_predictions_best*.npy               # 下面三个仅在 save_test_predictions: true（默认）时生成
       model_predictions_best*_thresholds.json   # 该 npy 的 test 阈值报告（每次覆盖，§5.3）
       thresholds_history.jsonl                  # 每次落 npy 追加一行，跨 epoch 阈值漂移留痕
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
| `fusion_temporal_injection` | `Models.fusion_temporal_injection` | `TemporalInjectionGATGRU` | 方案 3（TI）：`s_t` 与 `W_t` 作用后的线性残差相加再进 BiGRU，`u_t = s_t + g ⊙ (s_t W_t + A_t (s_t W_t))`，`g` 零初始化（`residual_injection: true`，默认）；`false` 时退回 `u_t = act(s_t W_t + A_t (s_t W_t))` | 748,929 |
| `fusion_film` | `Models.fusion_film` | `FiLMGATGRU` | 方案 4（FiLM）：只由 `W_t` 生成一对调制参数 `s'_t = (1+γ_t)⊙s_t + β_t` | 847,617 |

共同约定（改动前请连读 §4.4）：

- **复用不复制**：`MatGRUCell_PyG` 直接 import 自 `Models/egcn.py`，`precompute_adj_matrices` 取自 `Models/common_vectorized.py`。EvolveGCN-H 的实现若变化会自动传导到这两个模型。
- **冻结前缀对齐**：`feature_extractor` / `static_encoder` 进 `backbone.pth`；**算子演化（MatGRU）+ GRU 全部塞进 `temporal_encoder`**，因此 Phase 2 会把方案 3/4 的全部时序部件一起重初始化。若把 MatGRU 放到别处，frozen 模式的数字就与 GAT-GRU 不可比。
- **forward 签名**：与 §4.2 的"常规"签名相同（`forward(link_indices, current_times)`），trainer 无需改动。
- **恒等初值（两个模型都必须有）**：`fusion_film` 靠 `summarizer.mlp` 末层零初始化 ⇒ 初值 `γ=β=0`、`s' ≡ s`（自检 `_tmp_dump/test_fusion_models.py` 实测 `max|s'−s| = 0`）；`fusion_temporal_injection` 靠 `injection_gate` 零初始化 ⇒ 初值 `u_t ≡ s_t`，与 GAT-GRU 逐位一致（自检 `_tmp_dump/probe_ti_init.py` 实测 eval 下 `max|p_TI−p_GAT-GRU| = 0`）。**没有恒等初值的新架构一律先坏**：TI 原始形式 `act(...)` 在真实图上把 97.4% 的注入值压成精确 0、2014 年后整段恒为 0，epoch 1 即塌成"全部样本同侧"（val AUC 0.4723 / P 0.5000 / R 1.0000 / Loss 1.53 > ln 2）。
- **`summary_mode`**（仅 `fusion_film`）：`weights`（默认，`g_t = mean_dim0(W_t)`，走 MatGRU，+0.163 M）或 `attention`（`g_t = mean_N(s_t)`，几乎零成本，靠逐年 `A_t` 变化获得时间性）。**不提供 `mean_N(z)` 摘要**——共享编码器逐年使用同一个 `z`，该摘要对 `t` 恒定，会使"逐时间步调制"失效。
- **`residual_injection`**（仅 `fusion_temporal_injection`，默认 `true`）：残差式注入 + 每通道零初始化门 `g`。**残差路径刻意不加 `activation`**（`u_t = s_t + g ⊙ (s_t W_t + A_t (s_t W_t))`），因为该投影在真实特征尺度（`x` std ≈ 0.087）下均值≈0、量级 ≈1e-4，套 ReLU 会 97.4% 归零并同时掐断 `g` 与 `W_t` 的梯度；非线性交给下游 BiGRU。`g=0` 不使 `∂L/∂g` 为零 ⇒ 门能自行打开。`false` 则保留原 `act` 形式，仅供对照/消融（复现塌陷数字用）。
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
- 因此"同一参数听谁的"由**键路径**决定，不是文件优先级：不同键路径即使同名也会**共存**且被不同消费方读取（历史案例：顶层 `trainer.patience` 与 `trainer.kwargs.patience` 曾同时存在、只有前者被读，已于 2026-09-12 收敛为单键，见 §8 陷阱 2）。
- 所有读取模型配置的脚本都走 `load_config`（旧版 4 处直接 `yaml.safe_load(Models/configs/*.yaml)` 已统一）；
  `Prediction/get_imputation_fast.load_imputation_config` 是等价的 inline 三段合并（common → model → `imputation_common.yaml`）。

### 2.2 键归属

| 键 | 定义处 | 消费方 | 说明 |
|---|---|---|---|
| `dataset.*` | common（模型可覆盖） | `create_dataloaders` | batch size、负采样开关、特征开关 |
| `dataset.use_attr_onehot` / `onehot_attrs` | common | `dataset_feature_kwargs` | 决定 X 的宽度 |
| `model.kwargs.*` | common（共享编码器）+ 模型 yaml | `resolve_auto_kwargs` → 模型构造 | 共享编码器参数**只有 common 一处** |
| `trainer.num_epochs` | common（**模型 yaml 不应覆盖**） | `DynamicGraphTrainer.train()`、`SEALTrainer.train()` | epoch **上限**（仅保底终止）；早停预算的唯一旋钮是 `trainer.selection.patience`（§5.3、§5.7）。旧顶层 `trainer.patience` 已于 2026-09-12 删除、不再被读取 |
| `trainer.kwargs.margin_lambda` | common | `DynamicGraphTrainer.__init__` | BCE + Margin 权重 |
| `trainer.kwargs.margin` | common | `DynamicGraphTrainer.__init__` | 排序损失要求的分数差（默认 `1.0`）；**所有头部以 Sigmoid 结尾 ⇒ 分数 ∈ [0,1]、可达最大差恰为 1.0**，故 `≥1.0` 只能靠饱和满足（见 §5.1） |
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
    selection_cfg=None,              # 选模/早停准则，见 §5.3（trainer.selection）
    lr_schedule_cfg=None,            # 学习率轨迹，见 §5.4（trainer.lr_schedule）
    save_test_predictions=True,      # 落 test 预测与阈值记录，见 §5.3 末条
)
```

- `val_loader=None` ⇒ `selection_split='test'`（旧协议回退）。
- `static_data` 仅供 Temp-SEAL 使用；签名不匹配但未传静态图会立刻抛错。
- 只有当 `factset_edges` 与 `node_mapping` 同时存在时才会计算 FactSet 统计。
- 未传 `selection_cfg`/`lr_schedule_cfg` ⇒ **静默回旧协议**（0.5×FSQ+0.5×AUC、lr 恒 1e-3）；7 个入口都会从 config 透传，勿裸调。
- `save_test_predictions=False` ⇒ `train()` 既不写 `model_predictions_best*.npy`，也不写配套的阈值记录（默认 `true`，与历史行为一致；见 §7）。
- **`margin`（默认 `1.0`）的语义与值域约束**：排序损失为 `max(0, margin - (p_pos - p_neg))`，即"正样本分数必须比负样本高出的差"。模型头部一律以 `Sigmoid` 结尾 ⇒ 分数 ∈ `[0, 1]`、**可达最大差恰为 `1.0`**。于是 `margin ≥ 1.0` **只有在端点 `(p_pos=1, p_neg=0)` 才成立**，hinge 永不关闭、对每个样本对持续施加"往两端推"的恒定梯度 ⇒ 分数极化（正样本贴 1、可分离负样本贴 0）、概率校准变差（Brier 上升）、难负样本厚尾时最优阈值被挤进 `[0.9, 1.0]` 窄缝并跨 run 漂移（`results/ANALYSIS_0911_run_vs_paper.md` 同批诊断）。要做真正的"排序间隔"消融，把 `margin` 调到 `0.25` 量级，或改为在 **logit** 上做 margin（头部去掉 Sigmoid）。
- `margin` 与 `margin_lambda` 都随 ckpt 落盘（见 §5.5），`run_sampling` 写入 `summary.yaml._meta.loss`：**不同 margin 的 run 不可混算**。

### 5.2 `train()`

```python
train(num_epochs, save_path='best_dynamic_graph_model.pth',
      patience=EARLY_STOP_PATIENCE_DEFAULT, max_factset_edges=None) -> (best_epoch, best_score)
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
- 最高分 ⇒ 覆盖写 `best_model.pth` 并把 `patience_counter` 清零；否则 `patience_counter += 1`，达到 `trainer.selection.patience` 即停——它是**唯一的早停旋钮**（2026-09-12 起旧顶层 `trainer.patience` 已删除且不再被读取；该键为空时回落代码常量 `EARLY_STOP_PATIENCE_DEFAULT = 10` 并打印告警）。
- `best_auc_model.pth` 是**旁路产物**：只按选模 split 的**原始 AUC** 取最高，不参与早停计数（供曲线/阈值脚本使用）。
- **test 阈值记录**（两个 trainer 的 `save_test_predictions_npy`）：每写一次 `model_predictions_best*.npy`，就对**刚写入的这批 test 分数**算一次 Youden's J 与 F1-max 阈值，打印到控制台并落两个文件——`<npy 名>_thresholds.json`（最新值，覆盖写）与同目录 `thresholds_history.jsonl`（每次追加一行，含 `tag`/`epoch`）。算法与键名和 `Prediction/find_threshold*.py` 完全一致（`utils.youden_f1max_thresholds`，实测逐位相同），**因此可与 `threshold_results.json` 直接比较**。注意三点口径：① 它描述的是**test** 分数，**不参与选模/早停**（那是选模 split 的事）；② F1-max 只在 `[0.1, 0.9)` 上细扫 step 0.001，**上界取不到**；③ 分数恒定（如 Temp-SEAL 退化）时 argmax J 落在 ROC 的 `+inf` 点，记录为 `youden_j=1.0` + `youden_j_at_inf=true`（避免写出 Infinity 字面量）。**开关**：顶层键 `save_test_predictions`（默认 `true`，见 §7）关掉时 `.npy` 与上述两个阈值文件**都不写**——守卫写在 `save_test_predictions_npy` 内部而非调用点，所有调用方（含将来的）自动遵守；`train()` 开头打印 `Save test predictions: on/off`。它只决定落不落盘，**不影响任何训练数字**。
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
- **早停 budget 必须大于调度 patience**，否则 lr 永远等不到衰减就被早停掐掉（`selection.patience: 15` vs `plateau_patience: 3`）。启动时会打印告警。
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
| 损失权重 | `margin_lambda`, `margin` —— BCE + `λ·MarginRankingLoss` 的协议口径（不同值的 run 不可比） |

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

- **训练轮次不再由模型配置限定**（2026-09-12）：`Models/configs/seal.yaml` 不再写 `trainer.num_epochs`（历史值 `2` 让 Temp-SEAL 严重欠训练），改为继承 `Training/common_config.yaml` 的 `trainer.num_epochs`（当前 60）——它只作为**保底上限**，实际停止由早停决定。`run_sampling.py -c seal -s seal` 走 `DynamicGraphTrainer`，读的正是同一个键。
- **早停已接通**：`train()` 统计"连续多少个 **epoch** 没有刷新选模 split 的 step 级 F1"（与 checkpoint 同一准则 ⇒ 二者不可能打架），达到 patience 即 `break`。patience 与 `DynamicGraphTrainer` **同源同口径**：唯一旋钮 `trainer.selection.patience`（当前 15），为空时回落 `EARLY_STOP_PATIENCE_DEFAULT = 10`（`train_dualseal.py` 以 `patience=None` 透传，构造函数默认值即该常量）；`seal.yaml` 不再有 `trainer.kwargs.patience` 这条影子键，顶层 `trainer.patience` 已于 2026-09-12 删除。`train()` 参数 `patience=0` 可关早停（此时必须给 `num_epochs`），`num_epochs=None` 表示不设上限。
- **产物**：`run_summary.json` 新增 `epochs_planned`（在役上限）、`patience`、`early_stopped`，且 `epochs` 由"计划轮次"改为**实际跑完的轮次**（`Analysis/visualize_results.py::read_recorded_epochs` 依赖它做 SEAL 曲线的时间轴换算，见 §6）。
- 仍只读 `trainer.kwargs.*` 里的 `lr / weight_decay / grad_clip_norm / val_ratio / lr_patience / lr_factor / periodic_eval_records / use_tensorboard`；`trainer.kwargs.margin / margin_lambda` 对它无效（无排序损失，只算 BCE）。
- 自带 `ReduceLROnPlateau(mode='max', factor=lr_factor, patience=lr_patience, min_lr)`，在 **step 级周期评估**处 `step(选模 split F1)` ⇒ `lr_patience` 单位是**评估次数**，不是 epoch（`train()` 在上限偏小时会打印"调度不会触发"告警）。

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
- SEAL 的 epoch 时长估算：SEAL 的 `Train/Epoch_*` 实际记在 batch-step 轴上，估算器需按其**实际训练轮次**换算；轮次由 `read_recorded_epochs(run_dir)` 从 `run_summary.json`（`epochs`）或 `summary.yaml`（`results.*.epochs_run` / `_meta.epochs_run`）读取，**不可再写死常数**（2026-09-12 前写死 `2`，与 `seal.yaml` 的 `num_epochs: 2` 绑定）；读不到时不换算，避免给出误导性的秒数。
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
| `trainer.num_epochs` | common（**模型不应覆盖**） | 两个 trainer | epoch **上限**，只保底终止；实际停止看早停（2026-09-12 起 `seal.yaml` 不再覆盖） |
| ~~`trainer.patience`~~（2026-09-12 删除） | — | — | 早停预算已收敛为单一旋钮 `trainer.selection.patience`；配置残留此键时 `Training/config_loader.py` 会点名告警 |
| `trainer.selection.*` | common | trainer | **选模/早停准则** |
| `trainer.lr_schedule.*` | common | trainer | 学习率轨迹 |
| `trainer.max_factset_edges` | common | FactSet 统计 | 监控指标方差 |
| `trainer.kwargs.margin_lambda` | common | trainer | BCE 与排序损失权衡 |
| `trainer.kwargs.margin` | common | trainer | 排序损失要求的**分数差**（默认 `1.0`）；Sigmoid 头部 ⇒ 分数 ∈ [0,1]、可达最大差 = 1.0，故 `≥1.0` 等价"要求饱和"，会**恶化概率校准并让最优阈值漂进 `[0.9,1.0]` 窄缝**；消融用 `0.25` 量级（§5.1） |
| `save_test_predictions` | common **顶层** | 7 个入口 → trainer | 是否落 test 预测 `.npy` **及** §5.3 的阈值记录；默认 `true`。`false` 时 `save_test_predictions_npy` 直接返回（不跑那段推理、不写 3 个文件），**不影响训练数字**；`train()` 开头打印 `Save test predictions: on/off`；模型 yaml 可覆盖（`load_config` 里模型侧胜出） |
| `pretrained.path` | common | 训练前 | 是否加载并冻结骨干 |
| `defaults.repeats`, `defaults.seed` / `--repeats`, `--seed` | `sample_setting.yaml` / CLI | `run_sampling.py` | 重复运行次数与种子计划；`>1` 时启用 `seed<SEED>/` 隔离布局（§5.8） |
| `defaults.rotate_donor` / `--rotate_donor`, `--no_rotate_donor` | `sample_setting.yaml` / CLI | `run_sampling.py` | 每 repeat 换一个 backbone 供体（`resolve_mode_order`）；`repeats==1` 无效，resume 时以 `config.yaml` 记录为准（§5.8.1） |
| `IMPUT_MULTI_RUN` / `--multi-run {longest,confidence}` | 环境变量 / CLI | `visualize_results.py`、`plot_train_4metrics.py` | 同 (model, flag) 多个 run 时的取舍；`confidence` = 画 mean ± std 带 |
| `--seed-ci` | CLI | `visualize_results.py` | 跨 run best-epoch 的 mean ± 95% CI 表 + CSV（隐含 `--multi-run confidence`） |
| `--results-dir`, `--out-dir`, `--metrics`, `--fig-metrics`, `--no-figure` | CLI | `Analysis/seed_ci_summary.py` | 多种子汇总的输入目录/指标/图 |

---

## 8. 已知陷阱

1. **`sample_setting.yaml → output.subdir` 写死 `gatgru`**：换 backbone 必须加 `--subdir <name>`，否则写进 GAT-GRU 的目录。
2. **同键模型赢 / 不同键共存**：同一参数写在不同键路径会**共存**且被不同消费方读取。**epoch 上限与早停预算是全局键**：只在 `Training/common_config.yaml` 维护，模型 yaml 不要覆盖（`seal.yaml` 已于 2026-09-12 去掉 `num_epochs: 2` 与 `kwargs.patience`）。
   - **早停只有一个键**（2026-09-12 收敛）：顶层 `trainer.patience` 已删除，`DynamicGraphTrainer` 与 `SEALTrainer` 都不再读它；唯一旋钮是 `trainer.selection.patience`（为空 ⇒ 代码常量 `EARLY_STOP_PATIENCE_DEFAULT = 10` + 启动告警）。残留的 `trainer.patience` / `trainer.kwargs.patience` 会被 `Training/config_loader.py::_warn_on_deprecated_keys` 在**每次 `load_config`** 时点名警告，因此"写了却不生效"不会再静默。
3. **`run_sampling.py` 恒用 `DynamicGraphTrainer`**：不读 config 的 `trainer.module/class`，所以"配置里写的 trainer 类型"不会生效。
4. **协议入口必须成组检查**：`run_sampling.py` 与 5 个 `train_*.py` 都会传 `trainer.selection.*` / `trainer.lr_schedule.*` / `trainer.kwargs.margin_lambda` / `trainer.kwargs.margin`（2026-09-12 起统一；margin 于 2026-09-12 接通）。
   - 新增入口时如果漏传这几个键，会**静默回到旧协议**（0.5×FSQ + 0.5×AUC、恒定 lr、`margin=1.0`），而日志里看不出区别。`DynamicGraphTrainer` 的默认值就是旧协议，这是有意的向后兼容设计。
   - `SEALTrainer`（`train_dualseal.py`）只读 `trainer.kwargs` 的 lr/调度类键、**只有 BCE**（无排序损失），自成一套（`trainer.kwargs.lr_factor/lr_patience`，单位是评估次数）；但 epoch 上限与早停 patience 与共享 trainer 同源（§5.7）。
5. **FactSet 抽样用全局随机源**（`random.sample`）⇒ 单次运行的该列不可与其它运行逐位比较。
6. **SEAL 的旧结果（`num_epochs: 2` 时代）不可与新的早停训练结果并列**：`seal.yaml` 不再限定轮次后 Temp-SEAL 会跑到早停（最长 60 epoch），训练量级完全不同；跨版本比较前先看 `run_summary.json` 的 `epochs` / `epochs_planned` / `early_stopped`。
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
| 2026-09-12 | **TI 恒等初值修复**：`fusion_temporal_injection` 原形式 `u_t = act(s_t W_t + A_t (s_t W_t))` 在真实图上塌陷（注入值 97.4% 为精确 0、2014 年后整段恒 0 ⇒ epoch 1 全部样本同侧，val AUC 0.4723 / P 0.5000 / R 1.0000 / Loss 1.53）⇒ 改为残差注入 + 每通道零初始化门 `u_t = s_t + g ⊙ (s_t W_t + A_t (s_t W_t))`，初值逐位等价 GAT-GRU（FiLM 零初始化 MLP 的同构做法） | `Models/fusion_temporal_injection.py`、`Models/configs/fusion_temporal_injection.yaml` | `TemporalInjectionEncoder`/`TemporalInjectionGATGRU` 新增 `residual_injection=True` 参数；新增 `injection_gate`（+128 参数，748,801 → 748,929）；残差路径**不加 `activation`**（投影均值≈0、量级 1e-4，ReLU 会归零并掐断 `g`/`W_t` 梯度）；`false` 保留原 `act` 形式供对照。自检 `_tmp_dump/probe_ti_init.py`（合成图）与 `_tmp_dump/probe_ti_real.py`（真实图 51,509 节点）：`max|p_TI−p_GAT-GRU| = 0`、GRU 输入 std 2.269e-02 且 0 个零 | §1.2, §7 |

| 2026-09-12 | **早停收敛为单旋钮**：删除顶层 `trainer.patience`（此前它与 `trainer.selection.patience` 并存，后者非空时前者是死键）；唯一旋钮 `trainer.selection.patience`（为空 ⇒ 代码常量 `EARLY_STOP_PATIENCE_DEFAULT = 10` + 启动告警）；新增 `config_loader` 废弃键告警 | `Training/common_config.yaml`、`Training/trainer_common.py`、`Training/config_loader.py`、`Training/train_dualseal.py`、`Training/train_{gatgru,gatgru_1dire,bigru,egcn,tna}.py`、`SampleSetting/run_sampling.py`、`Models/configs/seal.yaml` | 新增常量 `trainer_common.EARLY_STOP_PATIENCE_DEFAULT = 10`；`DynamicGraphTrainer.train(patience=...)` 默认值 `10` → 该常量（数值不变）；`SEALTrainer.__init__(patience=15)` → 该常量；7 个入口不再读 `trainer_cfg['patience']`；`config_loader.load_config()` 新增 `_warn_on_deprecated_keys()`（残留 `trainer.patience` / `trainer.kwargs.patience` 时点名告警，干净配置静默）。**训练数字不变**：`selection.patience: 15` 一直生效 ⇒ 与 09-12 之前的 run 仍可比 | §2.1, §2.2, §5.2, §5.3, §5.7, §7, §8 |
| 2026-09-12 | **落 test 预测时一并报告/记录阈值**：对刚写入 npy 的 test 分数算 Youden's J 与 F1-max，控制台打印 + 落 `<npy>_thresholds.json` 与 `thresholds_history.jsonl` | `utils.py`（新增 `youden_f1max_thresholds`、`F1_SCAN`）、`Training/trainer_common.py`、`Training/train_dualseal.py` | `save_test_predictions_npy` 签名新增可选参数 `epoch=None, tag=''`（旧调用无需改动）；新增方法 `_record_test_thresholds`；`utils.py` 新增纯函数（仅输出诊断，**不接入选模/早停**）；npy 内容与既有读取方（`PrROC*.py`）完全不变 | §0.2, §5.3, §7 |
| 2026-09-12 | **接通 `save_test_predictions` 开关**（此前是无人读取的死键） | `Training/common_config.yaml`、`Training/trainer_common.py`、`Training/train_dualseal.py`、`Training/train_{gatgru,gatgru_1dire,bigru,egcn,tna}.py`、`SampleSetting/run_sampling.py` | 两个 trainer 构造新增 `save_test_predictions=True`（默认 true ⇒ 行为不变）；守卫在 `save_test_predictions_npy` 内部，关掉即同时跳过 `.npy` 与阈值记录；7 个入口按 `cfg.get('save_test_predictions', True)` 透传；`train()` 前导打印新增一行 | §5.3, §7 |
| 2026-09-12 | **接通 `trainer.kwargs.margin`**（此前恒为构造默认 1.0，配置改不动） | `Training/common_config.yaml`、`Training/trainer_common.py`、`Training/train_{gatgru,gatgru_1dire,bigru,egcn,tna}.py`、`SampleSetting/run_sampling.py` | 新增配置键 `trainer.kwargs.margin`（默认 `1.0` ⇒ 行为不变）；6 个 DynamicGraphTrainer 入口按 `trainer.kwargs.margin` 透传；trainer 新增 `self.margin` 并写入两个 ckpt（`margin`/`margin_lambda`）、前导打印一行；`summary.yaml._meta` 新增 `loss.{margin_lambda,margin}`。**语义**：头部是 Sigmoid ⇒ 分数 ∈ [0,1]、可达最大差 = 1.0，`margin ≥ 1.0` 退化为"要求饱和"（打分极化 / 校准恶化 / 阈值漂移），消融用 `0.25` 量级 | §4, §5.1, §5.5, §7, §8 |
| 2026-09-12 | **Temp-SEAL 不再在模型配置里限定训练轮次，改由 common 的 patience 决定**：`seal.yaml` 删除 `trainer.num_epochs: 2` 与 `trainer.kwargs.patience: 15`（继承 `common_config` 的 `num_epochs: 60` 作保底上限）；`SEALTrainer.train(num_epochs=None, patience=None)` 接通 **epoch 级早停**（连续 N 个 epoch 未刷新 step 级选模 F1 即停，与 checkpoint 同一准则），patience 解析顺序 `trainer.selection.patience → trainer.patience` 与 `DynamicGraphTrainer` 完全一致 | `Models/configs/seal.yaml`、`Training/train_dualseal.py`、`Training/common_config.yaml`、`Analysis/visualize_results.py` | `SEALTrainer.train()` 签名 `num_epochs` 默认值 2 → `None`（不设上限），新增 `patience` 形参（None → 用 `__init__` 值，0 → 关早停）；新增属性 `early_stopped`，`early_stop_counter` 由占位变实数；`run_summary.json` 新增 `epochs_planned/patience/early_stopped` 且 `epochs` 改为实际轮次；`visualize_results.py` 新增 `read_recorded_epochs()` 并把写死的 `default_num_epochs = 2` 换成 run 自记录值（`compute_epoch_duration_from_walltime(..., num_epochs=)`） | §0.1, §2.2, §5.7, §6, §7, §8 |

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
