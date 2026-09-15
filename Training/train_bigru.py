"""
BiGRU single-run training script
Usage: python Training/train_bigru.py [--config bigru_vec] [--device cuda:0]
      If --config is omitted, defaults to Models/configs/bigru_vec.yaml (vectorized version)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import inspect
import numpy as np
from datetime import datetime
from Training.config_loader import load_config
from Training.trainer_common import DynamicGraphTrainer, EARLY_STOP_PATIENCE_DEFAULT
from utils import import_attr, resolve_auto_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BiGRU single-run training')
    parser.add_argument('--config', '-c', default='bigru_vec',
                        help='model config name (default: bigru_vec, vectorized version)')
    parser.add_argument('--device', '-d', default='auto',
                        help='device (auto/cuda/cpu)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device

    cfg = load_config(args.config)
    ds_cfg = cfg['dataset']
    model_cfg = cfg['model']
    trainer_cfg = cfg['trainer']

    dataset_module = ds_cfg['module']
    create_dataloaders_fn = import_attr(dataset_module, 'create_dataloaders')
    dataset_feature_kwargs = import_attr(dataset_module, 'dataset_feature_kwargs')
    ModelClass = import_attr(model_cfg['module'], model_cfg['class'])

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', cfg['output_subdir'])
    os.makedirs(save_dir, exist_ok=True)
    run_name = datetime.now().strftime("%m%d-%H%M")

    dynamic_data, train_loader, val_loader, test_loader, full_dataset = create_dataloaders_fn(
        negative_ratio=ds_cfg.get('negative_ratio', 1),
        batch_size=ds_cfg['batch_size'],
        train_ratio=ds_cfg.get('train_ratio', 0.8),
        toy_mode=cfg.get('toy_mode', False),
        filter_factset_neg=ds_cfg.get('filter_factset_neg', False),
        intra_industry_neg=ds_cfg.get('intra_industry_neg', True),
        use_pred_neg=ds_cfg.get('use_pred_neg', True),
        **dataset_feature_kwargs(ds_cfg),
    )

    auto_context = {
        'num_features': dynamic_data.x.size(1),
        'num_nodes': dynamic_data.num_nodes,
        'time_steps': sorted(dynamic_data.edge_time.unique().tolist()),
        'hidden_dims': dynamic_data.x.size(1),
        'device': device,
    }
    model_kwargs = resolve_auto_kwargs(model_cfg['kwargs'], auto_context)
    model_init_params = inspect.signature(ModelClass.__init__).parameters
    if 'dynamic_data' in model_init_params:
        model = ModelClass(dynamic_data=dynamic_data, **model_kwargs)
    else:
        model = ModelClass(**model_kwargs)

    # --- Pretrained model loading (optional) ---
    pretrained_cfg = cfg.get('pretrained', {})
    pretrained_path = pretrained_cfg.get('path') if pretrained_cfg else None
    if pretrained_path:
        from Data.company_dataset import load_pretrained_backbone, reinit_trainable_parts
        model = load_pretrained_backbone(model, pretrained_path, device)
        reinit_trainable_parts(model)
        model = model.to(device)
        print(f"[Pretrained] loaded and froze feature extractor from {pretrained_path}")

    factset_edges = getattr(full_dataset, 'factset_edges', [])
    node_mapping = getattr(full_dataset, 'node_mapping', None)
    reverse_node_mapping = getattr(full_dataset, 'reverse_node_mapping', None)

    trainer = DynamicGraphTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        device=device,
        use_tensorboard=trainer_cfg.get('kwargs', {}).get('use_tensorboard', True),
        log_dir=os.path.join(save_dir, run_name),
        margin_lambda=trainer_cfg.get('kwargs', {}).get('margin_lambda', 0.1),
        # Ranking-loss margin, config key trainer.kwargs.margin (default 1.0 = old behaviour);
        # keep it < 1.0 because the Sigmoid head caps the achievable score gap at 1.0.
        margin=trainer_cfg.get('kwargs', {}).get('margin', 1.0),
        # Label smoothing of the BCE term, config key trainer.kwargs.label_smoothing
        # (absent / 0 = off = historical hard labels); see TechnicalGuide.md 5.1.
        label_smoothing=trainer_cfg.get('kwargs', {}).get('label_smoothing', 0.0),
        factset_edges=factset_edges,
        node_mapping=node_mapping,
        reverse_node_mapping=reverse_node_mapping,
        # Unified selection criterion (trainer.selection) + learning-rate schedule
        # (trainer.lr_schedule): keeps this single-config entry point on exactly the same protocol
        # as SampleSetting/run_sampling.py, so their numbers stay comparable.
        selection_cfg=trainer_cfg.get('selection'),
        lr_schedule_cfg=trainer_cfg.get('lr_schedule'),
        # Reporting switch (top-level config key): False skips the test-prediction .npy dump and the
        # threshold records that go with it. Default True, same as SampleSetting/run_sampling.py.
        save_test_predictions=cfg.get('save_test_predictions', True),
    )

    train_kwargs = {
        'num_epochs': trainer_cfg.get('num_epochs', 50),
        'save_path': os.path.join(save_dir, run_name, 'best_model.pth'),
        # Fallback only: the early-stopping budget is trainer.selection.patience (resolved inside
        # train()); the deprecated trainer.patience is no longer read.
        'patience': EARLY_STOP_PATIENCE_DEFAULT,
    }
    if trainer_cfg.get('max_factset_edges') is not None:
        train_kwargs['max_factset_edges'] = trainer_cfg['max_factset_edges']
    trainer.train(**train_kwargs)

