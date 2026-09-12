"""
SampleSetting multi-config sampling-mode training script.

The four sampling modes (ftt/ftf/fff/tff, i.e. the 2x2 combination of
filter_factset_neg x intra_industry_neg, defined in sample_setting.yaml) share the
same frozen backbone, so the sampling scheme is the only varying factor.
Phase 1 trains the first mode from scratch and saves its static_encoder as the shared
backbone; Phase 2 loads that backbone and freezes it for the remaining modes, training
only temporal_encoder + edge_predictor.

Output structure (results/gatgru/sampling_{MMDD-HHMM}/):
    sample_setting.yaml, backbone.pth, summary.yaml,
    {idx}_{mode}_scratch/ (donor mode trained from scratch), {idx}_{mode}_frozen/ (frozen mode)

Multi-seed repeats (--repeats N, default 1): the whole run above is repeated N times with seeds
base, base+1, ... Each repeat gets its own subdirectory so that the (differently initialised)
backbones can never be mixed up, plus a run-level summary.yaml that aggregates the repeats:

    results/gatgru/sampling_{MMDD-HHMM}/
        config.yaml, sample_setting.yaml, summary.yaml   <- aggregate over repeats
        seed42/{backbone.pth, 0_ftt_scratch/, 1_ftf_frozen/, ..., summary.yaml}
        seed43/{backbone.pth, ...}
        seed44/{backbone.pth, ...}

Donor rotation (--rotate_donor, on by default as soon as --repeats > 1; --no_rotate_donor
switches back to the fixed order): the training order of the four modes is shifted left by one
every repeat -- the mode that entered first moves to the end -- so the mode that supplies the
shared backbone changes as well:

    repeat 1: [0] ftt scratch -> [1] ftf -> [2] fff -> [3] tff   (the three frozen on ftt)
    repeat 2: [1] ftf scratch -> [2] fff -> [3] tff -> [0] ftt   (the three frozen on ftf)
    repeat 3: [2] fff scratch -> ...      repeat 4: [3] tff scratch -> ...

Every mode is therefore fully trained (and donates its backbone) in exactly one repeat and is a
frozen recipient in the others, which removes the systematic advantage of the mode selected by
--backbone_donor. Because the scratch/frozen split moves between repeats, the run-level summary
reports the repeats twice: `aggregate` keeps the exact mode key (role included) and `by_mode`
pools the roles of each mode (1 scratch + N-1 frozen, the balanced donor-rotated average).

Analysis/seed_ci_summary.py turns these per-seed summaries into mean ± 95% CI tables (it groups
by mode, not by mode key, whenever the run rotates its donor), and
Analysis/plot_train_4metrics.py --multi-run confidence draws the per-epoch confidence bands.

Usage:
    python SampleSetting/run_sampling.py                          # use YAML defaults
    python SampleSetting/run_sampling.py --num_rnn_layers 2
    python SampleSetting/run_sampling.py --config gatgru_vec --backbone_donor 3
    python SampleSetting/run_sampling.py --repeats 5              # 5 seeds (42..46)
    python SampleSetting/run_sampling.py --repeats 5 --seed 100   # 5 seeds (100..104)
    python SampleSetting/run_sampling.py --repeats 5 --no_rotate_donor   # keep one fixed donor
    python SampleSetting/run_sampling.py --resume results/gatgru/sampling_<timestamp>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import inspect
import yaml
import torch
import numpy as np
from datetime import datetime

from Training.config_loader import load_config
from Training.trainer_common import DynamicGraphTrainer
from utils import import_attr, resolve_auto_kwargs
from Data.company_dataset import load_pretrained_backbone, reinit_trainable_parts


# Load the SampleSetting config file
_SETTING_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YAML = os.path.join(_SETTING_DIR, 'sample_setting_new.yaml')


def load_sample_setting(yaml_path: str = None) -> dict:
    """Load the SampleSetting YAML config"""
    path = yaml_path or _DEFAULT_YAML
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # Validate required fields
    if 'sampling_modes' not in cfg:
        raise ValueError("Missing 'sampling_modes' field in config file")
    return cfg


# Single training run

def run_single_training(
    config_name: str,
    device: str,
    base_cfg: dict,
    model_cfg: dict,
    trainer_cfg: dict,
    ds_cfg: dict,
    sampling_params: dict,
    save_dir: str,
    run_name: str,
    model_kwargs_override: dict = None,
    pretrained_backbone: str = None,
    backbone_save_path: str = None,
    seed: int = 42,
):
    """
    Run one complete model training.

    Args:
        config_name: model config name (e.g. 'gatgru_vec')
        device: device
        base_cfg: full config (base + model merged)
        model_cfg: model sub-config
        trainer_cfg: trainer sub-config (may contain bootstrap overrides)
        ds_cfg: dataset sub-config (may contain bootstrap overrides)
        sampling_params: negative-sampling parameter dict
        save_dir: output directory
        run_name: TensorBoard subdirectory name
        model_kwargs_override: override model kwargs (e.g. num_rnn_layers)
        pretrained_backbone: if provided, load and freeze static_encoder
        backbone_save_path: if provided, after training extract static_encoder from best_model and save here
        seed: random seed

    Returns:
        dict: {
            'best_epoch': int,
            'best_score': float,         # value of the unified selection criterion
            'selection_split': str,      # 'val' (standard) or 'test' (legacy fallback)
            'selection_criterion': str,  # e.g. 'EMA5(AUC)' or '0.5*EMA5(FSQ) + 0.5*EMA5(AUC)'
            'selection_score_raw': float,  # unsmoothed score of the selected epoch
            'val_loss'/'val_f1'/'val_auc': float,
            'test_loss'/'test_f1'/'test_auc': float,
            'train_bce': float,          # training BCE component at the selected epoch
            'factset_quantile': float,        # FactSet quantile of the selection split
            'factset_quantile_test': float,   # FactSet quantile of the test split
            'wasserstein_diff': float,
            'wasserstein_diff_test': float,
            'epoch_seconds': float,      # duration of the selected epoch
            'avg_epoch_seconds': float,  # mean epoch duration over the whole run
            'lr': float,                 # learning rate at the selected epoch
            'final_lr': float,           # learning rate at the end of the run
            'lr_events': list,           # [{'epoch', 'from', 'to'}, ...] real reductions
            'epochs_run': int,
            'selection_config'/'lr_schedule_config': dict,  # effective protocol of this run
        }
    """
    dataset_module = ds_cfg['module']
    create_dataloaders_fn = import_attr(dataset_module, 'create_dataloaders')
    dataset_feature_kwargs = import_attr(dataset_module, 'dataset_feature_kwargs')
    ModelClass = import_attr(model_cfg['module'], model_cfg['class'])

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    # --- Load data (using the current sampling params) ---
    # create_dataloaders returns a 5-tuple (static_data, train_loader, val_loader, test_loader,
    # full_dataset). The validation split drives checkpoint selection and early stopping through the
    # single unified criterion configured in trainer.selection (see trainer_common.selection_score);
    # the test split is only reported.
    dynamic_data, train_loader, val_loader, test_loader, full_dataset = create_dataloaders_fn(
        negative_ratio=ds_cfg.get('negative_ratio', 1),
        batch_size=ds_cfg['batch_size'],
        train_ratio=ds_cfg.get('train_ratio', 0.8),
        toy_mode=base_cfg.get('toy_mode', False),
        min_degree=ds_cfg.get('min_degree', 2),
        source_filter=ds_cfg.get('source_filter', 'semi'),
        filter_factset_neg=sampling_params['filter_factset_neg'],
        intra_industry_neg=sampling_params['intra_industry_neg'],
        use_pred_neg=sampling_params.get('use_pred_neg', True),
        **dataset_feature_kwargs(ds_cfg),
    )

    # --- Build model ---
    auto_context = {
        'num_features': dynamic_data.x.size(1),
        'num_nodes': dynamic_data.num_nodes,
        'time_steps': sorted(dynamic_data.edge_time.unique().tolist()),
        'hidden_dims': dynamic_data.x.size(1),
        'device': device,
    }
    model_kwargs = resolve_auto_kwargs(model_cfg['kwargs'], auto_context)
    if model_kwargs_override:
        model_kwargs.update(model_kwargs_override)

    model_init_params = inspect.signature(ModelClass.__init__).parameters
    if 'dynamic_data' in model_init_params:
        model = ModelClass(dynamic_data=dynamic_data, **model_kwargs)
    else:
        model = ModelClass(**model_kwargs)

    # Load pretrained backbone (Phase 2)
    if pretrained_backbone is not None:
        print(f"  Loading pretrained backbone: {pretrained_backbone}")
        model = load_pretrained_backbone(model, pretrained_backbone, device)
        model = model.to(device)
        reinit_trainable_parts(model)

    # Extract trainer params
    factset_edges = getattr(full_dataset, 'factset_edges', [])
    node_mapping = getattr(full_dataset, 'node_mapping', None)
    reverse_node_mapping = getattr(full_dataset, 'reverse_node_mapping', None)

    trainer_kwargs = trainer_cfg.get('kwargs', {})
    trainer = DynamicGraphTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        use_tensorboard=trainer_kwargs.get('use_tensorboard', True),
        log_dir=os.path.join(save_dir, run_name),
        margin_lambda=trainer_kwargs.get('margin_lambda', 0.1),
        factset_edges=factset_edges,
        node_mapping=node_mapping,
        reverse_node_mapping=reverse_node_mapping,
        # Unified selection criterion (trainer.selection) and learning-rate schedule
        # (trainer.lr_schedule) come from the merged config; omitting them reproduces the legacy
        # protocol (0.5 * FactSet quantile + 0.5 * val AUC, constant lr).
        selection_cfg=trainer_cfg.get('selection'),
        lr_schedule_cfg=trainer_cfg.get('lr_schedule'),
        # The graph returned by create_dataloaders is passed to the model (as dynamic_data) and to
        # the trainer (as static_data): Temp-SEAL extracts its enclosing subgraphs from it and
        # therefore has a forward(data, link_indices, current_times) signature. Ignored by the
        # backbones whose forward is forward(link_indices, current_times).
        static_data=dynamic_data,
    )

    train_kwargs = {
        'num_epochs': trainer_cfg.get('num_epochs', 50),
        'save_path': os.path.join(save_dir, 'best_model.pth'),
        'patience': trainer_cfg.get('patience', 10),
    }
    if trainer_cfg.get('max_factset_edges') is not None:
        train_kwargs['max_factset_edges'] = trainer_cfg['max_factset_edges']

    # Train (checkpoint selection / early stopping: the unified criterion in trainer.selection)
    best_epoch, best_score = trainer.train(**train_kwargs)

    # Save backbone (Phase 1 extracts static_encoder weights)
    if backbone_save_path is not None:
        # Load best_model's weights back into model, then extract static_encoder
        best_ckpt = torch.load(train_kwargs['save_path'], map_location='cpu', weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
        save_backbone(model, backbone_save_path)

    # Collect result metrics (read from the best_model checkpoint)
    ckpt = torch.load(train_kwargs['save_path'], map_location='cpu', weights_only=False) if \
        backbone_save_path is None else best_ckpt
    result = {
        'best_epoch': best_epoch,
        'best_score': best_score,
        'selection_split': ckpt.get('selection_split', None),
        'val_loss': ckpt.get('val_loss', None),
        'val_f1': ckpt.get('val_f1', None),
        'val_auc': ckpt.get('val_auc', None),
        'test_loss': ckpt.get('test_loss', None),
        'test_auc': ckpt.get('test_auc', None),
        'test_f1': ckpt.get('test_f1', None),
        'train_bce': ckpt.get('train_bce', None),
        'train_margin': ckpt.get('train_margin', None),
        # FactSet statistics of the selection split ...
        'factset_quantile': ckpt.get('factset_quantile', None),
        'wasserstein_pos': ckpt.get('wasserstein_pos', None),
        'wasserstein_neg': ckpt.get('wasserstein_neg', None),
        'wasserstein_diff': ckpt.get('wasserstein_diff', None),
        # ... and of the test split, which is what the paper reports. For runs produced before the
        # validation-based selection switch both coincide, hence the fallback.
        'factset_quantile_test': ckpt.get('factset_quantile_test', ckpt.get('factset_quantile', None)),
        'wasserstein_diff_test': ckpt.get('wasserstein_diff_test', ckpt.get('wasserstein_diff', None)),
        'epoch_seconds': ckpt.get('epoch_seconds', None),
        'avg_epoch_seconds': float(np.mean(trainer.epoch_durations)) if trainer.epoch_durations else None,
        # Criterion / schedule bookkeeping, so that a run is self-describing in summary.yaml
        'selection_criterion': ckpt.get('selection_criterion', trainer.selection_criterion),
        'selection_score_raw': ckpt.get('selection_score_raw', None),
        'lr': ckpt.get('lr', None),
        'final_lr': trainer._current_lr(),
        'lr_events': [dict(e) for e in trainer.lr_events],
        'epochs_run': len(trainer.epoch_durations),
        'selection_config': dict(trainer.selection_cfg),
        'lr_schedule_config': dict(trainer.lr_schedule_cfg),
    }
    return result


# Save backbone (static_encoder weights only)

def save_backbone(model, save_path: str):
    """Save static_encoder weights (compatible with load_pretrained_backbone's loading format)"""
    backbone_state = {k: v.cpu() for k, v in model.state_dict().items()
                      if k.startswith(('static_encoder.', 'static_gnn.', 'feature_extractor.'))}
    # Wrap with the 'model_state_dict' key; load_pretrained_backbone reads the weights
    # via checkpoint.get('model_state_dict', checkpoint)
    torch.save({'model_state_dict': backbone_state}, save_path)
    print(f"  Backbone saved: {save_path} ({len(backbone_state)} keys)")


def extract_backbone_from_checkpoint(checkpoint_path: str, save_path: str) -> bool:
    """Extract static_encoder weights directly from an existing best_model.pth into backbone.pth

    Used as a resume fallback: training may be interrupted before save_backbone(),
    leaving best_model.pth present but backbone.pth missing.
    """
    if not os.path.exists(checkpoint_path):
        return False
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    full_state = ckpt.get('model_state_dict', ckpt)
    backbone_state = {k: v.cpu() for k, v in full_state.items()
                      if k.startswith(('static_encoder.', 'static_gnn.', 'feature_extractor.'))}
    if not backbone_state:
        print(f"  Warning: no static_encoder key found in checkpoint, cannot extract backbone")
        return False
    torch.save({'model_state_dict': backbone_state}, save_path)
    print(f"  Extracted backbone from donor best_model: {save_path} ({len(backbone_state)} keys)")
    return True


# Resume helper functions

def _load_result_from_checkpoint(checkpoint_path: str) -> dict:
    """Extract training result metrics from best_model.pth"""
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    test_auc = ckpt.get('test_auc', None)
    val_auc = ckpt.get('val_auc', None)
    factset_q = ckpt.get('factset_quantile', None)

    # The score that produced this checkpoint is stored verbatim by the trainer, because the
    # criterion is configurable (trainer.selection) and can no longer be reconstructed from the
    # metrics alone. Legacy checkpoints carry no selection_score, so they are rebuilt with the old
    # formula: 0.5 * factset_quantile + 0.5 * AUC of the selection split. Runs produced before the
    # validation-based selection switch carry no val_auc and fall back to test_auc.
    select_auc = val_auc if val_auc is not None else test_auc
    best_score = ckpt.get('selection_score')
    if best_score is None:
        if factset_q is not None and select_auc is not None:
            best_score = 0.5 * factset_q + 0.5 * select_auc
        elif select_auc is not None:
            best_score = select_auc  # fallback: AUC only
        else:
            best_score = None

    return {
        'best_epoch': ckpt.get('epoch', '?'),
        'best_score': best_score,
        'selection_split': ckpt.get('selection_split', None),
        'selection_criterion': ckpt.get('selection_criterion', 'legacy-composite'),
        'selection_score_raw': ckpt.get('selection_score_raw', None),
        'lr': ckpt.get('lr', None),
        'val_loss': ckpt.get('val_loss', None),
        'val_f1': ckpt.get('val_f1', None),
        'val_auc': val_auc,
        'test_loss': ckpt.get('test_loss', None),
        'test_auc': test_auc,
        'test_f1': ckpt.get('test_f1', None),
        'train_bce': ckpt.get('train_bce', None),
        'train_margin': ckpt.get('train_margin', None),
        'factset_quantile': factset_q,
        'wasserstein_pos': ckpt.get('wasserstein_pos', None),
        'wasserstein_neg': ckpt.get('wasserstein_neg', None),
        'wasserstein_diff': ckpt.get('wasserstein_diff', None),
        'factset_quantile_test': ckpt.get('factset_quantile_test', factset_q),
        'wasserstein_diff_test': ckpt.get('wasserstein_diff_test', ckpt.get('wasserstein_diff', None)),
        'epoch_seconds': ckpt.get('epoch_seconds', None),
        'avg_epoch_seconds': None,
    }


def write_effective_config(base_output_dir: str, cfg: dict, run_info: dict,
                           sampling_modes: list, resume_timestamp=None) -> str:
    """Record the merged configuration (common_config <- model config) next to the run outputs.

    The effective values of a run are otherwise not recoverable: only the per-mode sampling
    switches and the results are saved. On resume the original config.yaml of the first run is
    kept intact and the resume is recorded separately as config_resume_{timestamp}.yaml.
    Returns the path of the written file.
    """
    record = {
        '_note': 'Auto-generated at run time. Effective config = Training/common_config.yaml '
                 'deep-merged with Models/configs/{model}.yaml (same-named model keys win).',
        'run': run_info,
        'sampling_modes': [
            {'index': i, 'name': m['name'],
             'filter_factset_neg': m.get('filter_factset_neg', False),
             'intra_industry_neg': m.get('intra_industry_neg', False),
             'use_pred_neg': m.get('use_pred_neg', False)}
            for i, m in enumerate(sampling_modes)
        ],
        'merged_config': cfg,
    }
    if resume_timestamp:
        run_info['resumed_at'] = resume_timestamp
        path = os.path.join(base_output_dir, f'config_resume_{resume_timestamp}.yaml')
    else:
        path = os.path.join(base_output_dir, 'config.yaml')
    os.makedirs(base_output_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(record, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return path


def _scan_completed_modes(base_output_dir: str, sampling_modes: list,
                          donor_idx: int) -> dict:
    """Scan existing directories and return the {mode_key: result} map of completed modes"""
    completed = {}
    # donor mode (scratch)
    donor_mode = sampling_modes[donor_idx]
    donor_dir = os.path.join(base_output_dir, f"{donor_idx}_{donor_mode['name']}_scratch")
    if os.path.exists(donor_dir):
        ckpt_path = os.path.join(donor_dir, 'best_model.pth')
        res = _load_result_from_checkpoint(ckpt_path)
        if res is not None:
            key = f"{donor_idx}_{donor_mode['name']}_scratch"
            completed[key] = res

    # Check frozen modes
    for i, m in enumerate(sampling_modes):
        if i == donor_idx:
            continue
        mode_dir = os.path.join(base_output_dir, f"{i}_{m['name']}_frozen")
        if os.path.exists(mode_dir):
            ckpt_path = os.path.join(mode_dir, 'best_model.pth')
            res = _load_result_from_checkpoint(ckpt_path)
            if res is not None:
                key = f"{i}_{m['name']}_frozen"
                completed[key] = res

    return completed


# Mode order / donor rotation

def resolve_mode_order(sampling_modes: list, donor_idx: int, repeat_idx: int,
                       rotate: bool = True) -> list:
    """Mode indices in training order for one repeat (element 0 is the backbone donor).

    Without rotation the order is fixed: the configured donor first, then the remaining modes in
    their YAML order. With rotation the sequence is shifted left by one every repeat, so the mode
    that entered first moves to the end and a different mode supplies the backbone each time --
    no mode is systematically the recipient of somebody else's backbone.
    """
    n = len(sampling_modes)
    if n == 0:
        return []
    anchor = list(range(donor_idx, n)) + list(range(0, donor_idx))
    if not rotate:
        return anchor
    k = int(repeat_idx) % n
    return anchor[k:] + anchor[:k]


def mode_label(mode_key: str) -> str:
    """'2_fff_frozen' -> '2_fff' (the sampling mode itself, regardless of the role it played)."""
    for suffix in ('_scratch', '_frozen'):
        if mode_key.endswith(suffix):
            return mode_key[:-len(suffix)]
    return mode_key


def _role_of(mode_key: str) -> str:
    """'scratch' (donor, all parameters trainable) or 'frozen' (backbone frozen)."""
    return 'scratch' if mode_key.endswith('_scratch') else 'frozen'


def _read_recorded_mode_orders(base_output_dir: str) -> tuple:
    """(rotate_donor, {seed: order}) recorded by the first run of this directory, if any.

    config.yaml is written before training starts, so a resumed run can recover the rotation it
    was launched with instead of guessing it from the current defaults (which would silently send
    the repeats to the wrong scratch/frozen directories).
    """
    path = os.path.join(base_output_dir, 'config.yaml')
    if not os.path.exists(path):
        return None, {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            run_info = (yaml.safe_load(f) or {}).get('run', {}) or {}
    except Exception:
        return None, {}
    orders = {}
    for seed, order in (run_info.get('mode_order') or {}).items():
        try:
            orders[int(seed)] = [int(i) for i in order]
        except (TypeError, ValueError):
            continue
    return run_info.get('rotate_donor'), orders


# Seed-ensemble helpers (repeated runs)

# Keys that are not scalars and therefore cannot be averaged across seeds
_SEED_METRIC_EXCLUDE = {'selection_config', 'lr_schedule_config', 'lr_events', 'seeds'}


def resolve_seed_plan(base_seed: int, repeats: int) -> list:
    """Seeds of the repeats: base_seed, base_seed + 1, ..., base_seed + repeats - 1.

    Consecutive seeds keep different seeds from different flags (and from the 42-hard-coded split)
    while staying easy to read in directory names.
    """
    repeats = max(1, int(repeats))
    return [int(base_seed) + k for k in range(repeats)]


def seed_run_dir(base_output_dir: str, seed: int, repeats: int) -> str:
    """Output directory of one repeat.

    repeats == 1 keeps the historical flat layout (the run root itself), so every existing script
    and analysis path is unaffected. With repeats > 1 each seed gets its own subdirectory, which is
    what keeps the per-seed backbones (and their checkpoints, event files and summaries) apart.
    """
    if int(repeats) <= 1:
        return base_output_dir
    return os.path.join(base_output_dir, f'seed{seed}')


def discover_seed_dirs(base_output_dir: str) -> list:
    """Seeds of the `seedNNN` subdirectories of a run root ([] for the single-run layout)."""
    if not os.path.isdir(base_output_dir):
        return []
    seeds = []
    for name in sorted(os.listdir(base_output_dir)):
        if name.startswith('seed') and name[4:].isdigit():
            seeds.append(int(name[4:]))
    return sorted(seeds)


def _seed_from_summary(base_output_dir: str):
    """Seed recorded in <dir>/summary.yaml by a previous (single-repeat) run, if any."""
    path = os.path.join(base_output_dir, 'summary.yaml')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta = (yaml.safe_load(f) or {}).get('_meta', {}) or {}
        seed = meta.get('seed')
        return int(seed) if seed is not None else None
    except Exception:
        return None


def _t95_quantile(n: int) -> float:
    """Two-sided 95% t quantile with n-1 degrees of freedom (small hard-coded table)."""
    if n <= 1:
        return float('nan')
    df = n - 1
    if df >= 31:
        return 1.96
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
             15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
             22: 2.074, 24: 2.064, 26: 2.056, 28: 2.048, 30: 2.042}
    if df in table:
        return table[df]
    smaller = [k for k in table if k <= df]
    return table[max(smaller)]


def _metric_stats(values) -> dict:
    """mean / std / 95% CI half-width / n of one metric over a small sample (t distribution)."""
    values = [float(v) for v in values
              if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not values:
        return None
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else None
    ci95 = float(_t95_quantile(n) * std / np.sqrt(n)) if (n > 1 and std is not None) else None
    return {'mean': mean, 'std': std, 'ci95': ci95, 'n': n, 'values': values}


def aggregate_seed_results(per_seed_results: dict) -> dict:
    """Aggregate per-repeat results across seeds.

    Args:
        per_seed_results: {seed: {mode_key: result_dict}} as returned by run_single_training.

    Returns:
        {mode_key: {metric: {'mean', 'std', 'ci95', 'n', 'values'}}}, where ci95 is the two-sided
        95% confidence half-width of the mean (t-distribution). std/ci95 are None with a single
        seed, because a single run carries no dispersion information.
    """
    mode_keys = []
    for res in per_seed_results.values():
        for key in res:
            if key not in mode_keys:
                mode_keys.append(key)

    aggregate = {}
    for key in mode_keys:
        metrics = {}
        names = []
        for seed in sorted(per_seed_results):
            for name, value in per_seed_results[seed].get(key, {}).items():
                if (name not in names and name not in _SEED_METRIC_EXCLUDE
                        and isinstance(value, (int, float)) and not isinstance(value, bool)):
                    names.append(name)
        for name in names:
            values = [per_seed_results[s].get(key, {}).get(name) for s in sorted(per_seed_results)]
            stats = _metric_stats(values)
            if stats is not None:
                metrics[name] = stats
        aggregate[key] = metrics
    return aggregate


def aggregate_by_mode(per_seed_results: dict) -> dict:
    """Aggregate the repeats per sampling mode, pooling the roles that mode played.

    With donor rotation each mode is trained from scratch (and donates the backbone) in exactly
    one repeat while being a frozen recipient in the others, so every mode carries the same role
    composition and the pooled mean can no longer favour whichever mode --backbone_donor picked.
    The price of that balance is that the pooled values mix the two training regimes, hence
    'by_role' keeps them apart (scratch = all parameters trainable, frozen = temporal encoder +
    edge predictor only) and 'seeds' records which repeat contributed which role.

    Returns:
        {mode_label: {'seeds': {'scratch': [...], 'frozen': [...]},
                      'metrics': {metric: {'mean', 'std', 'ci95', 'n', 'values'}},
                      'by_role': {'scratch': {metric: {...}}, 'frozen': {metric: {...}}}}}
    """
    pooled = {}
    for seed in sorted(per_seed_results):
        for key, res in per_seed_results[seed].items():
            label = mode_label(key)
            role = _role_of(key)
            slot = pooled.setdefault(label, {'seeds': {'scratch': [], 'frozen': []}, 'values': {}})
            slot['seeds'][role].append(int(seed))
            for name, value in res.items():
                if (name in _SEED_METRIC_EXCLUDE or isinstance(value, bool)
                        or not isinstance(value, (int, float))):
                    continue
                slot['values'].setdefault(name, {'scratch': [], 'frozen': []})[role].append(float(value))

    by_mode = {}
    for label, slot in pooled.items():
        metrics, by_role = {}, {}
        for name, per_role in slot['values'].items():
            stats = _metric_stats(per_role['scratch'] + per_role['frozen'])
            if stats is None:
                continue
            metrics[name] = stats
            by_role[name] = {role: _metric_stats(per_role[role]) for role in ('scratch', 'frozen')
                             if per_role[role]}
        by_mode[label] = {'seeds': slot['seeds'], 'metrics': metrics, 'by_role': by_role}
    return by_mode


def _format_by_mode_report(by_mode: dict) -> str:
    """Console table of the per-mode aggregates (role-pooled and frozen-only)."""
    lines = [f"  {'Mode':<10} {'TestAUC pooled':<21} {'TestAUC frozen':<21} "
             f"{'F1 pooled':<21} {'Q_test pooled':<21} {'n':<3} {'roles':<8}"]

    def _pm(entry):
        if not entry or entry.get('mean') is None:
            return 'N/A'
        if entry.get('ci95') is None:
            return f"{entry['mean']:.4f} (n={entry.get('n', 0)})"
        return f"{entry['mean']:.4f}±{entry['ci95']:.4f}"

    for label, entry in by_mode.items():
        metrics = entry['metrics']
        frozen = (entry['by_role'].get('test_auc') or {}).get('frozen')
        seeds = entry['seeds']
        roles = f"S{len(seeds['scratch'])}/F{len(seeds['frozen'])}"
        lines.append(f"  {label:<10} {_pm(metrics.get('test_auc')):<21} {_pm(frozen):<21} "
                     f"{_pm(metrics.get('test_f1')):<21} {_pm(metrics.get('factset_quantile_test')):<21} "
                     f"{metrics.get('test_auc', {}).get('n', 0):<3} {roles:<8}")
    return '\n'.join(lines)


def _format_rotation_matrix(per_seed_results: dict) -> str:
    """Per-repeat table: which role each mode played, and its test AUC.

    Makes the donor rotation auditable at a glance -- every row (mode) is the donor exactly once
    (S), so the balanced design can be checked without opening the per-seed summaries.
    """
    seeds = sorted(per_seed_results)
    labels = []
    for seed in seeds:
        for key in per_seed_results[seed]:
            if mode_label(key) not in labels:
                labels.append(mode_label(key))
    labels.sort(key=lambda lab: (int(lab.split('_')[0]) if lab.split('_')[0].isdigit() else 0, lab))

    donor_row = []
    for seed in seeds:
        scratch = [k for k in per_seed_results[seed] if k.endswith('_scratch')]
        donor_row.append(scratch[0].split('_')[1] if scratch else '?')

    lines = ["  Role per repeat: S = trained from scratch and donated the backbone, "
             "F = frozen on that repeat's backbone",
             f"  {'Mode':<10}" + ''.join(f" {f'seed{s}':<17}" for s in seeds)]
    for label in labels:
        cells = []
        for seed in seeds:
            hit = [k for k in per_seed_results[seed] if mode_label(k) == label]
            if not hit:
                cells.append(f"{'-':<17}")
                continue
            key = hit[0]
            value = per_seed_results[seed][key].get('test_auc')
            cells.append(f"{(f'{_role_of(key)[0].upper()} ' + (f'{value:.4f}' if value is not None else 'N/A')):<17}")
        lines.append(f"  {label:<10}" + ''.join(f" {c}" for c in cells))
    lines.append(f"  {'Donor':<10}" + ''.join(f" {d:<17}" for d in donor_row))
    return '\n'.join(lines)


def _format_aggregate_report(aggregate: dict) -> str:
    """Console table of the seed-aggregated metrics (mean ± 95% CI)."""
    lines = []
    lines.append(f"  {'Mode':<20} {'Epoch':<12} {'TestAUC':<19} {'F1':<19} "
                 f"{'Q_test':<19} {'W_test':<19} {'n':<3}")
    lines.append(f"  {'-'*118}")

    def _pm(entry):
        if entry is None:
            return 'N/A'
        if entry['ci95'] is None:
            return f"{entry['mean']:.4f}±n/a"
        return f"{entry['mean']:.4f}±{entry['ci95']:.4f}"

    for key, metrics in aggregate.items():
        lines.append(
            f"  {key:<20} {_pm(metrics.get('best_epoch')):<12} {_pm(metrics.get('test_auc')):<19} "
            f"{_pm(metrics.get('test_f1')):<19} {_pm(metrics.get('factset_quantile_test')):<19} "
            f"{_pm(metrics.get('wasserstein_diff_test')):<19} "
            f"{metrics.get('test_auc', {}).get('n', 0):<3}")
    return '\n'.join(lines)


# Parse arguments

def parse_args(sample_cfg: dict = None):
    """Parse command-line arguments (preferring YAML defaults)"""
    defaults = sample_cfg.get('defaults', {}) if sample_cfg else {}
    default_config = defaults.get('config', 'gatgru_vec')
    default_device = defaults.get('device', 'auto')
    default_donor = defaults.get('backbone_donor', 0)
    default_repeats = defaults.get('repeats', 1)
    default_seed = defaults.get('seed', 42)

    parser = argparse.ArgumentParser(
        description='SampleSetting: 4 negative-sampling modes x shared-backbone training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Config file: SampleSetting/sample_setting.yaml (sampling modes, default params)
Model config: Models/configs/{default_config}.yaml, Training/common_config.yaml

Examples:
  python SampleSetting/run_sampling.py                          # use YAML defaults
  python SampleSetting/run_sampling.py --num_rnn_layers 2       # 2-layer BiGRU setting (see Appendix A.3)
  python SampleSetting/run_sampling.py --backbone_donor 3       # use tff as the backbone donor
  python SampleSetting/run_sampling.py --repeats 5               # seed-ensemble: seeds {default_seed}..{int(default_seed)+4}
  python SampleSetting/run_sampling.py --repeats 4 --no_rotate_donor   # same donor in every repeat
  python SampleSetting/run_sampling.py -c gatgru_vec -d cuda:0
        """
    )
    parser.add_argument('--config', '-c', default=default_config,
                        help=f'model config name (default: {default_config})')
    parser.add_argument('--subdir', '-s', default=None,
                        help='results subdirectory (default: output.subdir in sample_setting.yaml); '
                             'set it to keep another backbone out of the default folder, '
                             'e.g. --subdir seal')
    parser.add_argument('--device', '-d', default=default_device,
                        help=f'device (auto/cuda/cpu, default: {default_device})')
    parser.add_argument('--num_rnn_layers', type=int, default=None,
                        help='override num_rnn_layers (default: YAML model_overrides or model yaml config)')
    parser.add_argument('--backbone_donor', type=int, default=default_donor,
                        help=f'backbone donor mode index (default: {default_donor})')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from an existing output directory path (overrides the resume setting in YAML)')
    parser.add_argument('--repeats', type=int, default=None,
                        help=f'number of repeated runs with different seeds (default: {default_repeats}); '
                             'each repeat re-initialises the model and saves its own backbone.pth')
    parser.add_argument('--seed', type=int, default=None,
                        help=f'first random seed; repeat k uses seed+k (default: {default_seed})')
    parser.add_argument('--rotate_donor', dest='rotate_donor', action='store_true', default=None,
                        help='shift the mode order by one every repeat so that each mode supplies '
                             'the backbone once (default: defaults.rotate_donor in '
                             'sample_setting.yaml, i.e. on as soon as --repeats > 1)')
    parser.add_argument('--no_rotate_donor', dest='rotate_donor', action='store_false', default=None,
                        help='keep the same backbone donor in every repeat (historical behaviour)')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the config only, do not run training')
    return parser.parse_args()


# Main flow

def main():
    # Load the SampleSetting YAML config
    sample_cfg = load_sample_setting()
    sampling_modes = sample_cfg['sampling_modes']

    # Parse command-line arguments (using YAML defaults)
    args = parse_args(sample_cfg)
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device

    # Resume: YAML resume or command-line --resume
    yaml_resume = sample_cfg.get('defaults', {}).get('resume')
    resume_dir = args.resume or (yaml_resume if yaml_resume and yaml_resume != 'null' else None)

    # Load model config
    cfg = load_config(args.config)
    ds_cfg = cfg['dataset']
    model_cfg = cfg['model']
    trainer_cfg = cfg['trainer']

    # Model param overrides (YAML model_overrides + command-line overrides)
    model_kwargs_override = {}
    yaml_overrides = sample_cfg.get('model_overrides', {}) or {}
    model_kwargs_override.update({k: v for k, v in yaml_overrides.items() if v is not None})
    if args.num_rnn_layers is not None:
        model_kwargs_override['num_rnn_layers'] = args.num_rnn_layers

    # Determine the backbone donor
    donor_idx = args.backbone_donor
    if donor_idx < 0 or donor_idx >= len(sampling_modes):
        print(f"backbone_donor must be between 0 and {len(sampling_modes)-1}, current: {donor_idx}")
        sys.exit(1)

    donor_mode = sampling_modes[donor_idx]
    frozen_modes = [m for i, m in enumerate(sampling_modes) if i != donor_idx]

    # Output directory
    output_cfg = sample_cfg.get('output', {})
    results_root = output_cfg.get('root', 'results')
    # The YAML value wins over the config name (existing runs live in results/gatgru), so switching
    # --config without --subdir would write another backbone into the GAT-GRU folder.
    results_subdir = args.subdir or output_cfg.get('subdir', args.config)
    dir_prefix = output_cfg.get('prefix', 'sampling')
    timestamp = datetime.now().strftime("%m%d-%H%M")
    rnn_suffix = f"_rnn{args.num_rnn_layers}" if args.num_rnn_layers else ""

    if resume_dir:
        # Resume mode: use the existing directory
        if not os.path.isabs(resume_dir):
            resume_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                resume_dir
            )
        if not os.path.exists(resume_dir):
            print(f"Resume directory does not exist: {resume_dir}")
            sys.exit(1)
        base_output_dir = resume_dir
        # Regenerate timestamp to record the resume time
        resume_timestamp = datetime.now().strftime("%m%d-%H%M")
    else:
        run_dir_name = f"{dir_prefix}_{timestamp}{rnn_suffix}"
        base_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            results_root, results_subdir, run_dir_name
        )

    # ---- Seed plan (repeated runs) ----
    defaults_cfg = sample_cfg.get('defaults', {}) or {}
    base_seed = args.seed if args.seed is not None else int(defaults_cfg.get('seed', 42))
    requested_repeats = args.repeats if args.repeats is not None else int(defaults_cfg.get('repeats', 1))
    existing_seeds = discover_seed_dirs(base_output_dir)
    if resume_dir:
        if existing_seeds:
            # Seed-ensemble run: repeat exactly the seeds already on disk
            seeds = existing_seeds
            if args.repeats is not None or args.seed is not None:
                print(f"  [Resume] --repeats/--seed ignored: resuming the "
                      f"{len(seeds)} seeds already in this run")
        else:
            # Historical single-run layout: prefer the seed recorded in its own summary
            recorded = _seed_from_summary(base_output_dir)
            seeds = [recorded if recorded is not None else base_seed]
    else:
        seeds = resolve_seed_plan(base_seed, requested_repeats)
    repeats = len(seeds)

    # ---- Donor rotation ----
    # The training order of the modes is shifted left by one per repeat, so the mode that enters
    # first (and therefore supplies the shared backbone) changes every repeat. Precedence:
    # command line > the rotation recorded by the first run (config.yaml, so that a resumed run
    # stays on the rotation it was launched with) > sample_setting.yaml (default: on for repeats > 1).
    recorded_rotate, recorded_orders = (None, {})
    if resume_dir:
        recorded_rotate, recorded_orders = _read_recorded_mode_orders(base_output_dir)
    if args.rotate_donor is not None:
        rotate_donor = bool(args.rotate_donor)
        if resume_dir and recorded_rotate is not None and bool(recorded_rotate) != rotate_donor:
            print(f"  [Resume] WARNING: donor rotation overridden on the command line "
                  f"({bool(recorded_rotate)} -> {rotate_donor}); the scratch/frozen directories "
                  f"recomputed for the repeats may not match the existing ones")
    elif recorded_rotate is not None:
        rotate_donor = bool(recorded_rotate)
    else:
        rotate_donor = bool(defaults_cfg.get('rotate_donor', True))

    mode_orders = {}
    for k, seed in enumerate(seeds):
        mode_orders[seed] = (recorded_orders.get(seed) if resume_dir else None) or \
            resolve_mode_order(sampling_modes, donor_idx, k, rotate_donor)

    # Print config
    print(f"\n{'='*60}")
    print(f"SampleSetting multi-config sampling training")
    if resume_dir:
        print(f"  [Resume mode] directory: {base_output_dir}")
    print(f"  Model: {args.config} ({cfg.get('name', args.config)})")
    print(f"  Device: {device}")
    print(f"  GRU layers: {model_kwargs_override.get('num_rnn_layers', 'yaml default')}")
    print(f"  Backbone donor{'' if repeats < 2 or not rotate_donor else ' (repeat 1)'}: "
          f"[{donor_idx}] {donor_mode['desc']}")
    print(f"  Output dir: {base_output_dir}")
    if repeats > 1:
        print(f"  Repeats: {repeats} (seeds: {', '.join(str(s) for s in seeds)}) "
              f"-> one seed*/ subdirectory each, with its own backbone.pth")
        if rotate_donor:
            print(f"  Donor rotation: ON -- the mode order shifts by one every repeat, so every "
                  f"mode supplies the backbone exactly once")
            for k, seed in enumerate(seeds):
                order = mode_orders[seed]
                chain = ' -> '.join(f"[{i}]{sampling_modes[i]['name']}" for i in order)
                print(f"    repeat {k + 1} (seed {seed}): {chain}")
                print(f"      donor (scratch): [{order[0]}] {sampling_modes[order[0]]['name']}"
                      f" | frozen on it: "
                      f"{', '.join(sampling_modes[i]['name'] for i in order[1:])}")
        else:
            print(f"  Donor rotation: OFF -- every repeat uses the same backbone donor")
    else:
        print(f"  Seed: {seeds[0]} (single run; use --repeats N for a seed ensemble)")
    print(f"  Effective trainer: num_epochs={trainer_cfg.get('num_epochs')}, "
          f"patience={trainer_cfg.get('patience')}, "
          f"max_factset_edges={trainer_cfg.get('max_factset_edges')}, "
          f"margin_lambda={trainer_cfg.get('kwargs', {}).get('margin_lambda')}")
    sel_cfg = trainer_cfg.get('selection') or {}
    lr_cfg = trainer_cfg.get('lr_schedule') or {}
    print(f"  Selection: use_factset={sel_cfg.get('use_factset', True)} "
          f"(FSQ is still logged when false), metric={sel_cfg.get('metric', 'auc')}, "
          f"ema_span={sel_cfg.get('ema_span', 1)}, "
          f"patience={sel_cfg.get('patience') or trainer_cfg.get('patience')}")
    print(f"  LR schedule: enabled={lr_cfg.get('enabled', False)}, name={lr_cfg.get('name', 'none')}, "
          f"base_lr={lr_cfg.get('base_lr', 0.001)}, min_lr={lr_cfg.get('min_lr')}, "
          f"warmup_epochs={lr_cfg.get('warmup_epochs', 0)}, total_epochs={lr_cfg.get('total_epochs', 0)}")
    print(f"  Merged config: Training/common_config.yaml <- Models/configs/{args.config}.yaml "
          f"(same-named model keys win); the shared trainer is always used "
          f"(Training.trainer_common.DynamicGraphTrainer)")
    print(f"{'='*60}")

    # Metadata shared by every repeat; each repeat adds its own seed, and the run root adds the
    # seed list. The selection criterion and the learning-rate schedule define the protocol of the
    # run: numbers produced under different settings are not directly comparable.
    base_meta = {
        'created': timestamp,
        'model_config': args.config,
        'model_name': cfg.get('name', args.config),
        'device': device,
        'model_overrides': model_kwargs_override or 'none',
        'backbone_donor': donor_idx,
        # Mode order per repeat (element 0 = the donor that supplies this repeat's backbone), so
        # that a repeat can always be traced back to the role each mode played in it.
        'rotate_donor': bool(rotate_donor),
        'mode_order': {str(s): list(mode_orders[s]) for s in seeds},
        'selection': dict(sel_cfg),
        'lr_schedule': dict(lr_cfg),
        'repeats': repeats,
        'seeds': list(seeds),
    }
    if resume_dir:
        base_meta['resumed_at'] = resume_timestamp

    # Record the effective configuration next to the outputs, before training starts, so that a
    # crashed or interrupted run still leaves an auditable copy of every effective parameter.
    # Skipped on --dry_run, which must stay free of side effects.
    if args.dry_run:
        print("  Effective config: (dry run, not written)")
    else:
        run_info = {
            'config_name': args.config,
            'model_name': cfg.get('name', args.config),
            'device': device,
            'backbone_donor': donor_idx,
            'backbone_donor_mode': donor_mode['name'],
            'rotate_donor': bool(rotate_donor),
            'mode_order': {str(s): list(mode_orders[s]) for s in seeds},
            'model_overrides': model_kwargs_override or 'none',
            'results_subdir': results_subdir,
            'output_dir': base_output_dir,
            'created': timestamp,
            'repeats': repeats,
            'seeds': list(seeds),
        }
        config_record_path = write_effective_config(
            base_output_dir, cfg, run_info, sampling_modes,
            resume_timestamp=resume_timestamp if resume_dir else None)
        print(f"  Effective config: {config_record_path}")
        # One copy of the sampling plan covers every repeat (each repeat has its own seed/backbone)
        with open(os.path.join(base_output_dir, 'sample_setting.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(sample_cfg, f, allow_unicode=True, default_flow_style=False)

    table_header = f"  {'#':<3} {'Mode':<6} {'Type':<9} {'Description':<55}"
    print(f"\n{'-'*80}")
    print(table_header)
    print(f"{'-'*80}")
    print(f"  {donor_idx:<3} {donor_mode['name']:<6} {'scratch':<9} {donor_mode['desc']:<55}")
    for i, m in enumerate(frozen_modes):
        idx = sampling_modes.index(m)
        print(f"  {idx:<3} {m['name']:<6} {'frozen':<9} {m['desc']:<55}")
    print(f"{'-'*80}")
    if repeats > 1 and rotate_donor:
        print(f"  Note: the table above describes repeat 1 -- with donor rotation the roles move to "
              f"the next mode every repeat (see the repeat plan above).")

    if args.dry_run:
        print("\n[Dry Run] Showing config only, not running training.")
        for seed in seeds:
            order = mode_orders[seed]
            print(f"    seed {seed} -> {seed_run_dir(base_output_dir, seed, repeats)}"
                  f"  (donor: [{order[0]}] {sampling_modes[order[0]]['name']})")
        return

    # ---- Repeated runs (seed ensemble) ----
    # repeats == 1 keeps the historical layout: _run_one_repeat writes straight into base_output_dir.
    # With repeats > 1 every repeat is trained inside its own seed*/ directory, which is what keeps
    # the per-seed backbones, checkpoints, event files and summaries from ever being mixed up.
    per_seed_results = {}
    for repeat_idx, seed in enumerate(seeds):
        run_dir = seed_run_dir(base_output_dir, seed, repeats)
        os.makedirs(run_dir, exist_ok=True)
        if repeats > 1:
            print(f"\n{'#'*70}")
            print(f"# Repeat {repeat_idx + 1}/{repeats}  (seed={seed})")
            print(f"#   {run_dir}")
            if rotate_donor:
                order = mode_orders[seed]
                print(f"#   donor: [{order[0]}] {sampling_modes[order[0]]['name']} (scratch)"
                      f" | order: {' -> '.join(sampling_modes[i]['name'] for i in order)}")
            print(f"{'#'*70}")
        per_seed_results[seed] = _run_one_repeat(
            ctx=dict(args=args, device=device, cfg=cfg, model_cfg=model_cfg,
                     trainer_cfg=trainer_cfg, ds_cfg=ds_cfg, sampling_modes=sampling_modes,
                     donor_idx=donor_idx, donor_mode=donor_mode, frozen_modes=frozen_modes,
                     mode_order=mode_orders[seed],
                     resume_dir=resume_dir,
                     resume_timestamp=resume_timestamp if resume_dir else None,
                     base_meta=base_meta, model_kwargs_override=model_kwargs_override),
            seed=seed, run_dir=run_dir, backbone_path=os.path.join(run_dir, 'backbone.pth'))

    # Run-level view over the repeats (each repeat also wrote its own summary.yaml)
    if repeats > 1:
        aggregate = aggregate_seed_results(per_seed_results)
        by_mode = aggregate_by_mode(per_seed_results)
        print(f"\n{'='*120}")
        print(f"  Seed-ensemble summary: {repeats} repeats, seeds {', '.join(str(s) for s in seeds)}")
        print(f"  Values are mean ± 95% CI across repeats (t-distribution); "
              f"per-repeat details in seed*/summary.yaml")
        print(f"{'='*120}")
        print(_format_aggregate_report(aggregate))
        if rotate_donor:
            print(f"\n  {_format_rotation_matrix(per_seed_results)}")
            print(f"\n  Every mode is the donor once and a frozen recipient {repeats - 1} time(s), "
                  f"so the four modes share the same roles and the comparison between them is no "
                  f"longer tied to which mode happened to supply the backbone.")
        print(f"\n  Per-mode aggregate"
              + (f" (roles pooled: 1 scratch + {repeats - 1} frozen per mode)"
                 if rotate_donor else
                 f" (no rotation: every repeat keeps the same role for a given mode)")
              + f" -- 'pooled' averages every occurrence of the mode, 'frozen' only its frozen runs")
        print(_format_by_mode_report(by_mode))
        if repeats < 5:
            print(f"\n  Note: {repeats} repeat(s) can show a spread, but 5+ seeds are needed before "
                  f"quoting a confidence interval in the paper.")

        seed_dirs = {str(s): os.path.basename(seed_run_dir(base_output_dir, s, repeats))
                     for s in seeds}
        group_summary = {
            '_meta': {**base_meta, 'output_dir': base_output_dir, 'seed_dirs': seed_dirs,
                      'donor_plan': {str(s): sampling_modes[mode_orders[s][0]]['name'] for s in seeds}},
            'aggregate': aggregate,
            'by_mode': by_mode,
        }
        with open(os.path.join(base_output_dir, 'summary.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(group_summary, f, allow_unicode=True, default_flow_style=False)
        print(f"\n  Run root: {base_output_dir}")
        print(f"  Aggregate summary: {os.path.join(base_output_dir, 'summary.yaml')}")
        for s in seeds:
            print(f"    seed {s}: "
                  f"{os.path.join(seed_run_dir(base_output_dir, s, repeats), 'summary.yaml')}")


def _run_one_repeat(ctx, seed, run_dir, backbone_path):
    """Train one complete repeat (Phase 1 donor + Phase 2 frozen modes) inside ``run_dir``.

    Every repeat is self-contained: its own backbone.pth, mode directories, event files and
    summary.yaml. ``seed`` is forwarded to run_single_training, which re-seeds torch/numpy and
    re-initialises the model, the negative sampling and the data loaders, so the repeats are
    independent initialisations rather than the same trajectory evaluated twice.
    """
    args = ctx['args']
    device = ctx['device']
    cfg = ctx['cfg']
    model_cfg = ctx['model_cfg']
    trainer_cfg = ctx['trainer_cfg']
    ds_cfg = ctx['ds_cfg']
    sampling_modes = ctx['sampling_modes']
    donor_idx = ctx['donor_idx']
    donor_mode = ctx['donor_mode']
    frozen_modes = ctx['frozen_modes']
    # Donor rotation: the mode order of this repeat decides who supplies the backbone and in which
    # order the remaining modes are trained on it. Falls back to the run-level donor when absent.
    mode_order = ctx.get('mode_order')
    if mode_order:
        donor_idx = int(mode_order[0])
        donor_mode = sampling_modes[donor_idx]
        frozen_modes = [sampling_modes[i] for i in mode_order[1:]]
    resume_dir = ctx['resume_dir']
    resume_timestamp = ctx['resume_timestamp']
    base_meta = ctx['base_meta']
    model_kwargs_override = ctx['model_kwargs_override']

    # Resume: scan completed modes
    all_results = {}
    if resume_dir:
        completed = _scan_completed_modes(run_dir, sampling_modes, donor_idx)
        all_results.update(completed)
        if completed:
            print(f"\nResume check: {len(completed)}/4 modes already completed, will skip")
            for key, r in completed.items():
                auc_str = f"AUC={r['test_auc']:.4f}" if r['test_auc'] is not None else ""
                print(f"     {key:<28} epoch={r['best_epoch']} {auc_str}")
        remaining = len(sampling_modes) - len(completed)
        if remaining == 0:
            print(f"\n  All modes completed, updating summary.yaml only")
        else:
            print(f"  {remaining} modes remaining to train\n")

    # Phase 1: train the backbone donor (from scratch, all params trainable)
    donor_key = f"{donor_idx}_{donor_mode['name']}_scratch"
    donor_dir = os.path.join(run_dir, f"{donor_idx}_{donor_mode['name']}_scratch")

    if donor_key in all_results:
        # Resume: Phase 1 already done, skip
        print(f"\n  Phase 1 skipped: [{donor_idx}] {donor_mode['name']} (best_model.pth already exists)")
        donor_result = all_results[donor_key]
        # Fallback: backbone.pth may not have been saved due to an interrupted run; extract from donor best_model
        if not os.path.exists(backbone_path):
            donor_best = os.path.join(donor_dir, 'best_model.pth')
            if os.path.exists(donor_best):
                if not extract_backbone_from_checkpoint(donor_best, backbone_path):
                    print(f"  Failed to extract backbone, Phase 2 will fail")
            else:
                print(f"  donor best_model not found: {donor_best}")
    else:
        print(f"\n{'='*60}")
        print(f"Phase 1: train Backbone Donor [{donor_idx}] {donor_mode['name']} (all params)")
        print(f"{'='*60}")

        donor_result = run_single_training(
            config_name=args.config,
            seed=seed,
            device=device,
            base_cfg=cfg,
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            ds_cfg=ds_cfg,
            sampling_params=donor_mode,
            save_dir=donor_dir,
            run_name='scratch_backbone',
            model_kwargs_override=model_kwargs_override,
            pretrained_backbone=None,
            backbone_save_path=backbone_path,
        )
        all_results[donor_key] = donor_result

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"
    print(f"\n  Phase 1 result: epoch={donor_result['best_epoch']}, "
          f"score={_fmt(donor_result['best_score'])}, "
          f"ValAUC={_fmt(donor_result['val_auc'])}, "
          f"TestAUC={_fmt(donor_result['test_auc'])}"
          + (f", Factset_Q(sel)={donor_result['factset_quantile']:.4f}" if donor_result['factset_quantile'] is not None else "")
          + (f", Factset_Q(test)={donor_result['factset_quantile_test']:.4f}" if donor_result['factset_quantile_test'] is not None else ""))

    # Phase 2: freeze the backbone and train the remaining 3 modes
    # Filter out already-completed frozen modes
    remaining_frozen = [m for i, m in enumerate(sampling_modes)
                        if i != donor_idx and f"{i}_{m['name']}_frozen" not in all_results]

    if remaining_frozen:
        print(f"\n{'='*60}")
        print(f"Phase 2: freeze the backbone, train the remaining {len(remaining_frozen)}/{len(sampling_modes)-1} modes")
        print(f"{'='*60}")

    for m in frozen_modes:
        idx = sampling_modes.index(m)
        frozen_key = f"{idx}_{m['name']}_frozen"
        mode_dir = os.path.join(run_dir, f"{idx}_{m['name']}_frozen")

        if frozen_key in all_results:
            # Resume: this mode is already done
            mode_result = all_results[frozen_key]
            auc_s = _fmt(mode_result['test_auc'])
            q_s = f", Factset_Q={mode_result['factset_quantile']:.4f}" if mode_result['factset_quantile'] is not None else ""
            print(f"\n  Mode [{idx}] {m['name']}: best_model.pth already exists, skipping")
            print(f"      result: epoch={mode_result['best_epoch']}, AUC={auc_s}{q_s}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Mode [{idx}] {m['name']}: {m['desc']}")
        print(f"{'─'*60}")

        mode_result = run_single_training(
            config_name=args.config,
            seed=seed,
            device=device,
            base_cfg=cfg,
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            ds_cfg=ds_cfg,
            sampling_params=m,
            save_dir=mode_dir,
            run_name=m['name'],
            model_kwargs_override=model_kwargs_override,
            pretrained_backbone=backbone_path,
        )
        all_results[frozen_key] = mode_result

        print(f"  result: epoch={mode_result['best_epoch']}, "
              f"score={mode_result['best_score']:.4f}, "
              f"ValAUC={_fmt(mode_result['val_auc'])}, "
              f"TestAUC={mode_result['test_auc']:.4f}" +
              (f", Factset_Q(sel)={mode_result['factset_quantile']:.4f}" if mode_result['factset_quantile'] else "") +
              (f", Factset_Q(test)={mode_result['factset_quantile_test']:.4f}" if mode_result['factset_quantile_test'] else ""))

    # Summary report
    print(f"\n{'='*80}")
    print(f"  Training complete — summary report")
    print(f"{'='*80}")
    header = (f"  {'Mode':<20} {'Type':<9} {'Epoch':<6} {'TestAUC':<8} "
              f"{'F1':<8} {'Q_test':<9} {'W_test':<9} {'Q_sel':<9} {'W_sel':<9} {'Sel':<4}")
    print(header)
    print(f"  {'-'*100}")

    for key, r in all_results.items():
        auc_str = f"{r['test_auc']:.4f}" if r['test_auc'] is not None else "N/A"
        f1_str = f"{r['test_f1']:.4f}" if r['test_f1'] is not None else "N/A"
        # Q_test/W_test are what the paper reports; Q_sel/W_sel are the monitored (validation) ones
        q_str = f"{r['factset_quantile_test']:.4f}" if r['factset_quantile_test'] is not None else "N/A"
        w_str = f"{r['wasserstein_diff_test']:.4f}" if r['wasserstein_diff_test'] is not None else "N/A"
        q_sel = f"{r['factset_quantile']:.4f}" if r['factset_quantile'] is not None else "N/A"
        w_sel = f"{r['wasserstein_diff']:.4f}" if r['wasserstein_diff'] is not None else "N/A"
        split = (r['selection_split'] or '?')[:3]

        print(f"  {key:<20} {'scratch' if '_scratch' in key else 'frozen':<9} "
              f"{r['best_epoch']:<6} {auc_str:<8} {f1_str:<8} {q_str:<9} {w_str:<9} "
              f"{q_sel:<9} {w_sel:<9} {split:<4}")

    print(f"\n  Output dir: {run_dir}")
    print(f"  Backbone of this repeat: {backbone_path}")

    # --- Save the summary of this repeat ---
    # Same content as a historical single run, plus the seed bookkeeping that tells the repeats of
    # a seed ensemble apart (Analysis/seed_ci_summary.py reads exactly these fields, including
    # selection_criterion, which is copied here so that seeds trained under different protocols
    # cannot be averaged together unnoticed).
    meta_info = {**base_meta, 'seed': seed, 'output_dir': run_dir,
                 'selection_criterion': donor_result.get('selection_criterion'),
                 # Which mode supplied this repeat's backbone, and which modes were trained frozen
                 # on it: with donor rotation these differ between the repeats of the same run.
                 'donor_mode': donor_mode['name'],
                 'frozen_modes': [m['name'] for m in frozen_modes]}
    summary_config = {
        '_meta': meta_info,
        'results': {
            key: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                  for kk, vv in r.items()}
            for key, r in all_results.items()
        }
    }
    with open(os.path.join(run_dir, 'summary.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(summary_config, f, allow_unicode=True, default_flow_style=False)
    print(f"  Summary: {os.path.join(run_dir, 'summary.yaml')}")
    return all_results


if __name__ == '__main__':
    main()
