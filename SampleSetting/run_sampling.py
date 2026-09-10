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

Usage:
    python SampleSetting/run_sampling.py                          # use YAML defaults
    python SampleSetting/run_sampling.py --num_rnn_layers 2
    python SampleSetting/run_sampling.py --config gatgru_vec --backbone_donor 3
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
_DEFAULT_YAML = os.path.join(_SETTING_DIR, 'sample_setting.yaml')


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
            'best_score': float,
            'test_auc': float,
            'test_f1': float,
            'factset_quantile': float,
            'wasserstein_diff': float,
        }
    """
    dataset_module = ds_cfg['module']
    create_dataloaders_fn = import_attr(dataset_module, 'create_dataloaders')
    ModelClass = import_attr(model_cfg['module'], model_cfg['class'])

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    # --- Load data (using the current sampling params) ---
    dynamic_data, train_loader, test_loader, full_dataset = create_dataloaders_fn(
        negative_ratio=ds_cfg.get('negative_ratio', 1),
        batch_size=ds_cfg['batch_size'],
        train_ratio=ds_cfg.get('train_ratio', 0.8),
        toy_mode=base_cfg.get('toy_mode', False),
        min_degree=ds_cfg.get('min_degree', 2),
        source_filter=ds_cfg.get('source_filter', 'semi'),
        filter_factset_neg=sampling_params['filter_factset_neg'],
        intra_industry_neg=sampling_params['intra_industry_neg'],
        use_pred_neg=sampling_params.get('use_pred_neg', True),
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
        test_loader=test_loader,
        device=device,
        use_tensorboard=trainer_kwargs.get('use_tensorboard', True),
        log_dir=os.path.join(save_dir, run_name),
        margin_lambda=trainer_kwargs.get('margin_lambda', 0.1),
        factset_edges=factset_edges,
        node_mapping=node_mapping,
        reverse_node_mapping=reverse_node_mapping,
    )

    train_kwargs = {
        'num_epochs': trainer_cfg.get('num_epochs', 50),
        'save_path': os.path.join(save_dir, 'best_model.pth'),
        'patience': trainer_cfg.get('patience', 10),
    }
    if trainer_cfg.get('max_factset_edges') is not None:
        train_kwargs['max_factset_edges'] = trainer_cfg['max_factset_edges']

    # Train (checkpoint selection: 0.5 x FactSet quantile + 0.5 x test AUC)
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
        'test_auc': ckpt.get('test_auc', None),
        'test_f1': ckpt.get('test_f1', None),
        'factset_quantile': ckpt.get('factset_quantile', None),
        'wasserstein_pos': ckpt.get('wasserstein_pos', None),
        'wasserstein_neg': ckpt.get('wasserstein_neg', None),
        'wasserstein_diff': ckpt.get('wasserstein_diff', None),
    }
    return result


# Save backbone (static_encoder weights only)

def save_backbone(model, save_path: str):
    """Save static_encoder weights (compatible with load_pretrained_backbone's loading format)"""
    backbone_state = {k: v.cpu() for k, v in model.state_dict().items()
                      if k.startswith('static_encoder.')}
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
                      if k.startswith('static_encoder.')}
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
    factset_q = ckpt.get('factset_quantile', None)

    # Composite score = 0.5 * factset_quantile + 0.5 * test_auc
    if factset_q is not None and test_auc is not None:
        best_score = 0.5 * factset_q + 0.5 * test_auc
    elif test_auc is not None:
        best_score = test_auc  # fallback: AUC only
    else:
        best_score = None

    return {
        'best_epoch': ckpt.get('epoch', '?'),
        'best_score': best_score,
        'test_auc': test_auc,
        'test_f1': ckpt.get('test_f1', None),
        'factset_quantile': factset_q,
        'wasserstein_pos': ckpt.get('wasserstein_pos', None),
        'wasserstein_neg': ckpt.get('wasserstein_neg', None),
        'wasserstein_diff': ckpt.get('wasserstein_diff', None),
    }


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


# Parse arguments

def parse_args(sample_cfg: dict = None):
    """Parse command-line arguments (preferring YAML defaults)"""
    defaults = sample_cfg.get('defaults', {}) if sample_cfg else {}
    default_config = defaults.get('config', 'gatgru_vec')
    default_device = defaults.get('device', 'auto')
    default_donor = defaults.get('backbone_donor', 0)

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
  python SampleSetting/run_sampling.py -c gatgru_vec -d cuda:0
        """
    )
    parser.add_argument('--config', '-c', default=default_config,
                        help=f'model config name (default: {default_config})')
    parser.add_argument('--device', '-d', default=default_device,
                        help=f'device (auto/cuda/cpu, default: {default_device})')
    parser.add_argument('--num_rnn_layers', type=int, default=None,
                        help='override num_rnn_layers (default: YAML model_overrides or model yaml config)')
    parser.add_argument('--backbone_donor', type=int, default=default_donor,
                        help=f'backbone donor mode index (default: {default_donor})')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from an existing output directory path (overrides the resume setting in YAML)')
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
    results_subdir = output_cfg.get('subdir', args.config)
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
        backbone_path = os.path.join(base_output_dir, 'backbone.pth')
        # Regenerate timestamp to record the resume time
        resume_timestamp = datetime.now().strftime("%m%d-%H%M")
    else:
        run_dir_name = f"{dir_prefix}_{timestamp}{rnn_suffix}"
        base_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            results_root, results_subdir, run_dir_name
        )
        backbone_path = os.path.join(base_output_dir, 'backbone.pth')

    # Print config
    print(f"\n{'='*60}")
    print(f"SampleSetting multi-config sampling training")
    if resume_dir:
        print(f"  [Resume mode] directory: {base_output_dir}")
    print(f"  Model: {args.config} ({cfg.get('name', args.config)})")
    print(f"  Device: {device}")
    print(f"  GRU layers: {model_kwargs_override.get('num_rnn_layers', 'yaml default')}")
    print(f"  Backbone donor: [{donor_idx}] {donor_mode['desc']}")
    print(f"  Output dir: {base_output_dir}")
    print(f"{'='*60}")

    table_header = f"  {'#':<3} {'Mode':<6} {'Type':<9} {'Description':<55}"
    print(f"\n{'-'*80}")
    print(table_header)
    print(f"{'-'*80}")
    print(f"  {donor_idx:<3} {donor_mode['name']:<6} {'scratch':<9} {donor_mode['desc']:<55}")
    for i, m in enumerate(frozen_modes):
        idx = sampling_modes.index(m)
        print(f"  {idx:<3} {m['name']:<6} {'frozen':<9} {m['desc']:<55}")
    print(f"{'-'*80}")

    if args.dry_run:
        print("\n[Dry Run] Showing config only, not running training.")
        return

    # Resume: scan completed modes
    all_results = {}
    if resume_dir:
        completed = _scan_completed_modes(base_output_dir, sampling_modes, donor_idx)
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
    donor_dir = os.path.join(base_output_dir, f"{donor_idx}_{donor_mode['name']}_scratch")

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
          f"AUC={_fmt(donor_result['test_auc'])}"
          + (f", Factset_Q={donor_result['factset_quantile']:.4f}" if donor_result['factset_quantile'] is not None else ""))

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
        mode_dir = os.path.join(base_output_dir, f"{idx}_{m['name']}_frozen")

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
              f"AUC={mode_result['test_auc']:.4f}" +
              (f", Factset_Q={mode_result['factset_quantile']:.4f}" if mode_result['factset_quantile'] else ""))

    # Summary report
    print(f"\n{'='*80}")
    print(f"  Training complete — summary report")
    print(f"{'='*80}")
    header = (f"  {'Mode':<20} {'Type':<9} {'Epoch':<6} {'AUC':<8} "
              f"{'F1':<8} {'Factset_Q':<10} {'W-Diff':<8}")
    print(header)
    print(f"  {'-'*76}")

    for key, r in all_results.items():
        auc_str = f"{r['test_auc']:.4f}" if r['test_auc'] is not None else "N/A"
        f1_str = f"{r['test_f1']:.4f}" if r['test_f1'] is not None else "N/A"
        q_str = f"{r['factset_quantile']:.4f}" if r['factset_quantile'] is not None else "N/A"
        w_str = f"{r['wasserstein_diff']:.4f}" if r['wasserstein_diff'] is not None else "N/A"

        print(f"  {key:<20} {'scratch' if '_scratch' in key else 'frozen':<9} "
              f"{r['best_epoch']:<6} {auc_str:<8} {f1_str:<8} {q_str:<10} {w_str:<8}")

    print(f"\n  Output dir: {base_output_dir}")
    print(f"  Shared backbone: {backbone_path}")

    # --- Save summary config ---
    meta_info = {
        'created': timestamp,
        'model_config': args.config,
        'model_name': cfg.get('name', args.config),
        'device': device,
        'model_overrides': model_kwargs_override or 'none',
        'backbone_donor': donor_idx,
        'output_dir': base_output_dir,
    }
    if resume_dir:
        meta_info['resumed_at'] = resume_timestamp
    summary_config = {
        '_meta': meta_info,
        'results': {
            key: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                  for kk, vv in r.items()}
            for key, r in all_results.items()
        }
    }
    with open(os.path.join(base_output_dir, 'summary.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(summary_config, f, allow_unicode=True, default_flow_style=False)
    # Save a copy of sample_setting.yaml into the output directory for reproducibility
    with open(os.path.join(base_output_dir, 'sample_setting.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(sample_cfg, f, allow_unicode=True, default_flow_style=False)
    print(f"  Summary: {os.path.join(base_output_dir, 'summary.yaml')}")
    print(f"  Config copy: {os.path.join(base_output_dir, 'sample_setting.yaml')}")


if __name__ == '__main__':
    main()
