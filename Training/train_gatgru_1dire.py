"""
GAT-GRU unidirectional temporal ablation training script (bidirectional ablation)
Usage: python Training/train_gatgru_1dire.py [--config gatgru_1dire] [--device cuda:0]
      If --config is omitted, defaults to Models/configs/gatgru_1dire.yaml
Reference: train the bidirectional version with python Training/train_gatgru.py --config gatgru_vec;
      the two versions differ only in temporal_encoder direction, everything else
      (data/hyperparameters/seed/training protocol) is identical.
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
from Training.trainer_common import DynamicGraphTrainer
from utils import import_attr, resolve_auto_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAT-GRU unidirectional temporal ablation training')
    parser.add_argument('--config', '-c', default='gatgru_1dire',
                        help='model config name (default: gatgru_1dire, unidirectional ablation version)')
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
    # The unidirectional ablation shares only static_encoder (GAT) with the bidirectional version;
    # the other layers have different dimensions, so only static_encoder.* is loaded and frozen,
    # while temporal_encoder/edge_predictor are reinitialized.
    pretrained_cfg = cfg.get('pretrained', {})
    pretrained_path = pretrained_cfg.get('path') if pretrained_cfg else None
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        model_state = checkpoint.get('model_state_dict', checkpoint)
        static_state = {k: v for k, v in model_state.items()
                        if k.startswith(('static_encoder.', 'feature_extractor.'))}
        if static_state:
            model.load_state_dict(static_state, strict=False)
            for p in model.static_encoder.parameters():
                p.requires_grad = False
            # The shared node-feature extractor is part of the frozen backbone as well
            if getattr(model, 'feature_extractor', None) is not None:
                for p in model.feature_extractor.parameters():
                    p.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"[Pretrained] loaded and froze static_encoder from {pretrained_path} "
                  f"({len(static_state)} tensors); temporal_encoder/edge_predictor reinitialized")
            print(f"[Pretrained] trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        else:
            print(f"[Pretrained] WARNING: no static_encoder.* weights found in checkpoint, skipping pretrained load")
        model = model.to(device)

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
        factset_edges=factset_edges,
        node_mapping=node_mapping,
        reverse_node_mapping=reverse_node_mapping,
    )

    train_kwargs = {
        'num_epochs': trainer_cfg.get('num_epochs', 50),
        'save_path': os.path.join(save_dir, run_name, 'best_model.pth'),
        'patience': trainer_cfg.get('patience', 10),
    }
    if trainer_cfg.get('max_factset_edges') is not None:
        train_kwargs['max_factset_edges'] = trainer_cfg['max_factset_edges']
    trainer.train(**train_kwargs)
