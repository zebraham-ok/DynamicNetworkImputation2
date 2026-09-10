"""
Bootstrap v2 training script -- structural stability evaluation of a model based on pretrained weights.

Each iteration: sample with replacement from the training pool -> dynamic negative
sampling -> reinitialize temporal_encoder + edge_predictor (freeze the static_encoder
backbone) -> train, with a different random seed per iteration.

Note: the Bootstrap data loader builds samples at 1:1 on the training set and at 1:2 on test/val.

Usage:
    python Bootstraps/bootstrap_v2_train.py
    python Bootstraps/bootstrap_v2_train.py -c Bootstraps/bootstrap_v2_config.yaml
    python Bootstraps/bootstrap_v2_train.py -c bootstrap_v2_config.yaml --pretrained path/to/other.pth --device cuda:1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import gc
import time
import inspect
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from utils import deep_merge, import_attr, resolve_auto_kwargs, resolve_device
from Data.company_dataset import (
    create_bootstrap_datasets,
    BootstrapIterationDataset,
    build_bootstrap_static_graph,
    load_pretrained_backbone,
    reinit_trainable_parts,
)


# Config loading

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BOOTSTRAP_CONFIG = os.path.join(ROOT_DIR, 'Bootstraps', 'bootstrap_v2_config.yaml')


def _load_yaml(path):
    """Load a YAML file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _dot_get(d, key_path, default=None):
    """Read a nested dict by dot path: _dot_get(cfg, 'training.lr')"""
    keys = key_path.split('.')
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def _dot_set(d, key_path, value):
    """Write a nested dict by dot path: _dot_set(cfg, 'training.epochs', 100)"""
    keys = key_path.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_model_config(config_name):
    """Load the model architecture config: Models/configs/{name}.yaml"""
    model_config_dir = os.path.join(ROOT_DIR, 'Models', 'configs')
    yaml_path = os.path.join(model_config_dir, f"{config_name}.yaml")
    if os.path.exists(yaml_path):
        return _load_yaml(yaml_path)
    elif os.path.exists(config_name):
        return _load_yaml(config_name)
    raise FileNotFoundError(f"Model config not found: {config_name}")


def merge_cli_overrides(bootstrap_cfg, cli_overrides):
    """Apply CLI overrides to the config dict"""
    for key_path, value in cli_overrides:
        if value is not None:
            # Cast according to the type of the existing yaml value
            existing = _dot_get(bootstrap_cfg, key_path)
            if existing is not None and not isinstance(existing, type(value)):
                try:
                    value = type(existing)(value)
                except (ValueError, TypeError):
                    pass
            _dot_set(bootstrap_cfg, key_path, value)
    return bootstrap_cfg


# Bootstrap trainer (trains only the trainable parts)

class BootstrapTrainer:
    """Bootstrap-specific trainer: trains only the trainable parts (temporal_encoder + edge_predictor)

    Supports two model architectures:
        - GAT-GRU/LSTM/TNA: forward(link_indices, current_times)
        - Temp-SEAL:          forward(dynamic_data, link_indices, current_times)
    """

    def __init__(self, model, train_loader, val_loader, device='cpu',
                 lr=0.001, weight_decay=1e-5, margin_lambda=0.1, margin=1.0,
                 log_dir=None, dynamic_data=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        # SEAL-like models need the full graph data as the first argument of forward
        self.dynamic_data = dynamic_data.to(device) if dynamic_data is not None else None
        self._is_seal = hasattr(model, 'subgraph_extractor')

        # Optimize only parameters with requires_grad=True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

        self.bce_criterion = nn.BCELoss()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.margin_lambda = margin_lambda

        # TensorBoard
        self.writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)

    def compute_metrics(self, predictions, labels, threshold=0.5):
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        if len(predictions) == 0:
            return 0.0, 0.0, 0.0, 0.0
        binary_preds = (predictions > threshold).float()
        labels_np = labels.cpu().numpy()
        preds_np = binary_preds.cpu().numpy()
        probs_np = predictions.cpu().numpy()
        try:
            precision = precision_score(labels_np, preds_np, zero_division=0)
            recall = recall_score(labels_np, preds_np, zero_division=0)
            f1 = f1_score(labels_np, preds_np, zero_division=0)
            auc = roc_auc_score(labels_np, probs_np) if len(np.unique(labels_np)) > 1 else 0.5
        except:
            precision = recall = f1 = auc = 0.0
        return precision, recall, f1, auc

    def train_epoch(self, epoch):
        self.model.train()
        total_bce = 0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for link_indices, current_times, labels in pbar:
            link_indices = link_indices.to(self.device)
            current_times = current_times.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            if self.dynamic_data is not None:
                predictions = self.model(self.dynamic_data, link_indices, current_times)
            else:
                predictions = self.model(link_indices, current_times)

            if len(predictions) <= 1:
                continue

            bce_loss = self.bce_criterion(predictions, labels)

            # Margin Ranking Loss
            margin_loss = torch.tensor(0.0, device=self.device)
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_scores = predictions[pos_mask]
                neg_scores = predictions[neg_mask]
                pos_expanded = pos_scores.repeat_interleave(len(neg_scores))
                neg_expanded = neg_scores.repeat(len(pos_scores))
                target = torch.ones_like(pos_expanded)
                margin_loss = self.margin_loss(pos_expanded, neg_expanded, target)

            loss = bce_loss + self.margin_lambda * margin_loss
            loss.backward()
            self.optimizer.step()

            total_bce += bce_loss.item()
            all_preds.append(predictions.detach())
            all_labels.append(labels.detach())

            pbar.set_postfix({'BCE': f'{bce_loss.item():.4f}'})

        if not all_preds:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        avg_bce = total_bce / len(self.train_loader)
        precision, recall, f1, auc = self.compute_metrics(all_preds, all_labels)

        if self.writer:
            self.writer.add_scalar('Train/Epoch_BCE', avg_bce, epoch)
            self.writer.add_scalar('Train/Epoch_F1', f1, epoch)
            self.writer.add_scalar('Train/Epoch_AUC', auc, epoch)

        return avg_bce, precision, recall, f1, auc

    @torch.no_grad()
    def evaluate(self, data_loader, epoch, mode='val'):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for link_indices, current_times, labels in data_loader:
            link_indices = link_indices.to(self.device)
            current_times = current_times.to(self.device)
            labels = labels.to(self.device).float()

            if self.dynamic_data is not None:
                predictions = self.model(self.dynamic_data, link_indices, current_times)
            else:
                predictions = self.model(link_indices, current_times)

            if len(predictions) <= 1:
                continue

            loss = self.bce_criterion(predictions, labels)
            total_loss += loss.item()
            all_preds.append(predictions)
            all_labels.append(labels)

        if not all_preds:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        avg_loss = total_loss / len(data_loader)
        precision, recall, f1, auc = self.compute_metrics(all_preds, all_labels)

        if self.writer:
            self.writer.add_scalar(f'{mode.capitalize()}/Epoch_Loss', avg_loss, epoch)
            self.writer.add_scalar(f'{mode.capitalize()}/Epoch_F1', f1, epoch)
            self.writer.add_scalar(f'{mode.capitalize()}/Epoch_AUC', auc, epoch)

        return avg_loss, precision, recall, f1, auc

    def train(self, num_epochs, save_path, patience=10):
        best_score = 0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            t0 = time.time()
            train_bce, train_prec, train_rec, train_f1, train_auc = self.train_epoch(epoch)
            val_loss, val_prec, val_rec, val_f1, val_auc = self.evaluate(
                self.val_loader, epoch, 'val'
            )
            duration = time.time() - t0

            print(f"  Epoch {epoch+1:3d} ({duration:.1f}s) | "
                  f"Train BCE={train_bce:.4f} F1={train_f1:.4f} AUC={train_auc:.4f} | "
                  f"Val Loss={val_loss:.4f} F1={val_f1:.4f} AUC={val_auc:.4f}")

            current_score = val_auc

            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'val_loss': val_loss,
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered! best epoch={best_epoch+1}, best val_auc={best_score:.4f}")
                    break

        if self.writer:
            self.writer.close()

        return best_epoch, best_score


# Main entry point

def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap v2: structural stability evaluation of a model based on pretrained weights'
    )
    parser.add_argument('-c', '--config', default=DEFAULT_BOOTSTRAP_CONFIG,
                        help=f'Bootstrap v2 config file path (default: {DEFAULT_BOOTSTRAP_CONFIG})')
    # CLI overrides (when the value is None, the value from the yml is used)
    parser.add_argument('--pretrained', default=None,
                        help='Override the pretrained weight path in the yml')
    parser.add_argument('--device', default=None,
                        help='Override the device setting in the yml')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the number of training epochs in the yml')
    parser.add_argument('--num_bootstrap', type=int, default=None,
                        help='Override the number of Bootstrap iterations in the yml')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override the batch size in the yml')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override the learning rate in the yml')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override the early-stopping patience in the yml')
    parser.add_argument('--output', '-o', default=None,
                        help='Override the output subdirectory in the yml')
    parser.add_argument('--seed_base', type=int, default=None,
                        help='Override the seed base in the yml')
    parser.add_argument('--toy_mode', action='store_true', default=None,
                        help='Force-enable toy mode')
    args = parser.parse_args()

    # Load the Bootstrap config (yml)
    print(f"\nLoading Bootstrap config: {args.config}")
    bootstrap_cfg = _load_yaml(args.config)

    # CLI overrides
    cli_overrides = [
        ('pretrained.path', args.pretrained),
        ('training.device', args.device),
        ('training.epochs', args.epochs),
        ('bootstrap.num_iterations', args.num_bootstrap),
        ('training.batch_size', args.batch_size),
        ('training.lr', args.lr),
        ('training.patience', args.patience),
        ('output.subdir', args.output),
        ('bootstrap.seed_base', args.seed_base),
        ('debug.toy_mode', args.toy_mode if args.toy_mode else None),
    ]
    bootstrap_cfg = merge_cli_overrides(bootstrap_cfg, cli_overrides)

    # Extract commonly used parameters (for variable-name compatibility)
    model_config_name = bootstrap_cfg['model']['config']
    pretrained_path = bootstrap_cfg['pretrained']['path']
    num_bootstrap = bootstrap_cfg['bootstrap']['num_iterations']
    seed_base = bootstrap_cfg['bootstrap']['seed_base']
    device = resolve_device(bootstrap_cfg['training']['device'])
    epochs = bootstrap_cfg['training']['epochs']
    patience = bootstrap_cfg['training']['patience']
    batch_size = bootstrap_cfg['training']['batch_size']
    lr = bootstrap_cfg['training']['lr']
    weight_decay = bootstrap_cfg['training'].get('weight_decay', 1e-5)
    margin_lambda = bootstrap_cfg['loss']['margin_lambda']
    margin = bootstrap_cfg['loss']['margin']
    toy_mode = bootstrap_cfg['debug'].get('toy_mode', False)

    # Data parameters
    data_cfg = bootstrap_cfg['data']
    output_cfg = bootstrap_cfg['output']

    # Load the model architecture config
    model_cfg = load_model_config(model_config_name)
    print(f"Model config: {model_config_name} ({model_cfg['name']})")

    # Output directory
    if output_cfg.get('subdir'):
        output_subdir = output_cfg['subdir']
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
        output_subdir = f"bootstrap_{model_cfg['output_subdir']}_{timestamp}"
    save_dir = os.path.join(ROOT_DIR, 'results', 'bootstrap', output_subdir)
    os.makedirs(save_dir, exist_ok=True)

    # Save the full run config (used by the analysis script)
    saved_config = dict(bootstrap_cfg)
    saved_config['_meta'] = {
        'model_config_name': model_config_name,
        'model_name': model_cfg['name'],
        'output_subdir': output_subdir,
        'save_dir': save_dir,
    }
    with open(os.path.join(save_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(saved_config, f, allow_unicode=True, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Bootstrap v2 Training")
    print(f"  Model config: {model_config_name} ({model_cfg['name']})")
    print(f"  Pretrained weights: {pretrained_path}")
    print(f"  Bootstrap iterations: B={num_bootstrap}")
    print(f"  Device: {device}")
    print(f"  Output directory: {save_dir}")
    print(f"{'='*60}\n")

    # 1. Three-way data split (fixed once and for all)
    print("[Step 1] Three-way data split "
          f"(Test={data_cfg['test_ratio']:.0%} / Val={data_cfg['val_ratio']:.0%} / "
          f"Train Pool={1-data_cfg['test_ratio']-data_cfg['val_ratio']:.0%})...")

    bootstrap_data = create_bootstrap_datasets(
        negative_ratio=data_cfg['negative_ratio'],
        embedding_name=data_cfg['embedding_name'],
        test_ratio=data_cfg['test_ratio'],
        val_ratio=data_cfg['val_ratio'],
        random_state=seed_base,
        min_degree=data_cfg.get('min_degree', 2),
        source_filter=data_cfg.get('source_filter', 'semi'),
        other_possible_fill=data_cfg.get('other_possible_fill', 0.0),
        filter_factset_neg=data_cfg.get('filter_factset_neg', False),
        intra_industry_neg=data_cfg.get('intra_industry_neg', True),
        toy_mode=toy_mode,
    )

    full_dataset = bootstrap_data['full_dataset']
    test_dataset = bootstrap_data['test_dataset']
    val_dataset = bootstrap_data['val_dataset']
    train_pool = bootstrap_data['train_pool']
    blacklist = bootstrap_data['blacklist']
    train_pool_dataset = bootstrap_data['train_pool_dataset']

    # Verify data isolation
    test_pos_set = set(tuple(s) for s in test_dataset.positive_samples)
    val_pos_set = set(tuple(s) for s in val_dataset.positive_samples)
    train_pool_set = set(train_pool)
    assert test_pos_set.isdisjoint(train_pool_set), \
        f"ERROR: {len(test_pos_set & train_pool_set)} test positive samples leaked into the training pool!"
    assert val_pos_set.isdisjoint(train_pool_set), \
        f"ERROR: {len(val_pos_set & train_pool_set)} val positive samples leaked into the training pool!"
    assert test_pos_set.isdisjoint(val_pos_set), \
        f"ERROR: {len(test_pos_set & val_pos_set)} test/val positive samples overlap!"
    print("  Data isolation check passed (Test/Val/TrainPool have no overlap)")

    # Build fixed test/val DataLoaders
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Build the static graph (compatible with the pretrained model)
    print("\n[Step 2] Building static graph...")

    dynamic_data = build_bootstrap_static_graph(bootstrap_data, None)
    torch.save(dynamic_data, os.path.join(save_dir, "dynamic_data.pyg"))
    print(f"  static graph: {dynamic_data.num_nodes} nodes, {dynamic_data.edge_index.shape[1]} edges")

    # 3. Create the model and load pretrained weights
    # If pretrained.path is unset or empty, run in from-scratch training mode
    scratch_mode = not pretrained_path

    if scratch_mode:
        print("\n[Step 3] From-scratch training mode (no pretrained weights)")
        print("  Round 1: train all parameters from random init -> extract static_encoder as the backbone for later rounds")
        print("  Round 2+: freeze the round-1 backbone, train only temporal_encoder + edge_predictor")
    else:
        print("\n[Step 3] Creating model and loading pretrained weights...")
        print(f"  Pretrained weights: {pretrained_path}")

    model_cfg_params = model_cfg['model']
    ModelClass = import_attr(model_cfg_params['module'], model_cfg_params['class'])

    # Use full_dataset's available_years as time_steps, covering all years
    available_years = sorted(bootstrap_data['full_dataset'].available_years)
    auto_context = {
        'num_features': dynamic_data.x.size(1),
        'num_nodes': dynamic_data.num_nodes,
        'time_steps': available_years,
        'hidden_dims': dynamic_data.x.size(1),
        'device': device,
    }
    model_kwargs = resolve_auto_kwargs(model_cfg_params['kwargs'], auto_context)

    model_init_params = inspect.signature(ModelClass.__init__).parameters
    if 'dynamic_data' in model_init_params:
        model = ModelClass(dynamic_data=dynamic_data, **model_kwargs)
    else:
        model = ModelClass(**model_kwargs)

    if not scratch_mode:
        model = load_pretrained_backbone(model, pretrained_path, device)
    model = model.to(device)

    # 4. Bootstrap iterative training
    print(f"\n[Step 4] Bootstrap iterative training (B={num_bootstrap})...")
    print(f"  Training set: fixed Val validation set (for early stopping)")
    if scratch_mode:
        print(f"  Mode: from scratch (round 1 trains all params -> later rounds freeze the backbone)")
    else:
        print(f"  Mode: pretrained backbone (reinitialize temporal_encoder + edge_predictor each iteration)")
    print(f"  Each iteration: sample with replacement from train_pool -> dynamic negative sampling -> train")
    print(f"{'='*60}")

    for b in range(num_bootstrap):
        seed = seed_base + b
        print(f"\n{'─'*60}")
        print(f"  Bootstrap iteration {b+1}/{num_bootstrap} (seed={seed})")
        print(f"{'─'*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- 4a. Sampling with replacement + dynamic negative sampling ---
        iter_dataset = BootstrapIterationDataset(
            train_pool=train_pool,
            blacklist=blacklist,
            meta_dataset=train_pool_dataset,
            negative_ratio=data_cfg['negative_ratio'],
            seed=seed,
            intra_industry_neg=data_cfg.get('intra_industry_neg', True),
        )

        # Verify data isolation
        iter_pos = set(iter_dataset.positive_samples)
        assert iter_pos.isdisjoint(test_pos_set), \
            f"ERROR: positive samples of bootstrap iter {b} overlap with the test set!"

        train_loader = PyGDataLoader(
            iter_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # --- 4b. Parameter initialization strategy ---
        if scratch_mode:
            if b == 0:
                # Round 1: all parameters randomly initialized and trainable
                print("  [Scratch] Round 1 from-scratch training (all parameters trainable)")
                for p in model.parameters():
                    p.requires_grad = True
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Round 2+: load and freeze the static_encoder from round 1's best_model, reinitialize the rest
                backbone_path = os.path.join(save_dir, 'scratch_backbone.pth')
                print(f"  [Scratch] Loading round-1 backbone: {backbone_path}")
                load_pretrained_backbone(model, backbone_path, device)
                model = model.to(device)
                reinit_trainable_parts(model)
        else:
            # Pretrained mode: reinitialize temporal_encoder + edge_predictor
            reinit_trainable_parts(model)

        # Print the number of trainable/frozen modules
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
        print(f"  Trainable modules: {len(set(n.split('.')[0] for n in trainable_names))}")
        print(f"  Frozen modules: {len(set(n.split('.')[0] for n in frozen_names))}")

        # --- 4c. Training ---
        log_dir = os.path.join(save_dir, 'runs', f'bootstrap_{b}')
        # SEAL-like models need the full dynamic_data as the first argument of forward
        seal_dynamic_data = dynamic_data if hasattr(model, 'subgraph_extractor') else None
        trainer = BootstrapTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            margin_lambda=margin_lambda,
            margin=margin,
            log_dir=log_dir,
            dynamic_data=seal_dynamic_data,
        )

        save_path = os.path.join(save_dir, f'best_model_{b}.pth')
        best_epoch, best_score = trainer.train(
            num_epochs=epochs,
            save_path=save_path,
            patience=patience,
        )

        # --- 4d. Record information ---
        print(f"  Iter {b+1} done | best_epoch={best_epoch+1}, best_val_auc={best_score:.4f}")

        # --- 4e. From-scratch mode: after round 1, extract the backbone for later rounds ---
        if scratch_mode and b == 0:
            backbone_path = os.path.join(save_dir, 'scratch_backbone.pth')
            best_ckpt = torch.load(save_path, map_location=device, weights_only=False)
            best_state = best_ckpt.get('model_state_dict', best_ckpt)
            backbone_prefix = 'static_encoder.'
            if hasattr(model, 'static_gnn') and not hasattr(model, 'static_encoder'):
                backbone_prefix = 'static_gnn.'
            backbone_state = {k: v for k, v in best_state.items()
                              if k.startswith(backbone_prefix)}
            torch.save({'model_state_dict': backbone_state}, backbone_path)
            print(f"  [Scratch] backbone extracted ({len(backbone_state)} keys) -> {backbone_path}")

        pd.DataFrame(iter_dataset.positive_samples, columns=["source", "target", "year"]).to_csv(
            os.path.join(save_dir, f"train_pool_{b}.csv"), index=False
        )

    # 5. Save metadata
    print(f"\n{'='*60}")
    print(f"Saving Bootstrap metadata...")

    pd.DataFrame(test_dataset.positive_samples, columns=["source", "target", "year"]).to_csv(
        os.path.join(save_dir, "test_pos.csv"), index=False
    )
    pd.DataFrame(val_dataset.positive_samples, columns=["source", "target", "year"]).to_csv(
        os.path.join(save_dir, "val_pos.csv"), index=False
    )
    pd.DataFrame(train_pool, columns=["source", "target", "year"]).to_csv(
        os.path.join(save_dir, "train_pool.csv"), index=False
    )
    pd.DataFrame(list(blacklist), columns=["source", "target", "year"]).to_csv(
        os.path.join(save_dir, "blacklist.csv"), index=False
    )

    with open(os.path.join(save_dir, 'split_info.yaml'), 'w') as f:
        yaml.dump({
            'total_positive': len(full_dataset.original_positive_samples),
            'test_positive': len(test_dataset.positive_samples),
            'val_positive': len(val_dataset.positive_samples),
            'train_pool_positive': len(train_pool),
            'blacklist_size': len(blacklist),
            'num_bootstrap': num_bootstrap,
        }, f)

    print(f"\nBootstrap v2 training fully complete!")
    print(f"   Output directory: {save_dir}")
    print(f"   Model files: best_model_0.pth ~ best_model_{num_bootstrap-1}.pth")
    print(f"   Next step: python Bootstraps/bootstrap_v2_analysis.py --dir {save_dir}")


if __name__ == "__main__":
    main()
