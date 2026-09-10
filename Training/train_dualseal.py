"""
SEAL trainer (step-level metrics version): a full evaluation every N batches.
Monitoring metrics (Loss/F1/AUC/BCE/Factset quantile/Wasserstein) are logged per batch step,
with metric naming aligned to trainer_common.py, to ease dual-axis display by step.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import wasserstein_distance
import numpy as np
from tqdm import tqdm
from Models.tempseal import SEALWithTemporalWeighting
from torch.utils.tensorboard import SummaryWriter


class SEALTrainer:
    def __init__(self, model: SEALWithTemporalWeighting, static_data, train_loader, test_loader,
                 device='cuda' if torch.cuda.is_available() else 'cpu', log_dir='runs/seal_experiment',
                 use_tensorboard=True,
                 lr=0.001, weight_decay=1e-4, grad_clip_norm=1.0,
                 val_ratio=0.1, patience=15, lr_patience=8, lr_factor=0.5,
                 factset_edges=None, node_mapping=None, reverse_node_mapping=None,
                 max_factset_edges=None,
                 periodic_eval_steps=1000,
                 periodic_eval_records=None):
        self.model = model.to(device)
        self.static_data = static_data.to(device)
        self.device = device

        # Factset external validation
        self.factset_edges = factset_edges if factset_edges else []
        self.node_mapping = node_mapping
        self.reverse_node_mapping = reverse_node_mapping
        self.max_factset_edges = max_factset_edges

        # Train/validation split
        if val_ratio > 0:
            train_size = int((1 - val_ratio) * len(train_loader.dataset))
            val_size = len(train_loader.dataset) - train_size
            train_subset, val_subset = random_split(
                train_loader.dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            self.train_loader = DataLoader(train_subset,
                batch_size=train_loader.batch_size, shuffle=True,
                num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory,
                persistent_workers=getattr(train_loader, 'persistent_workers', False))
            self.val_loader = DataLoader(val_subset,
                batch_size=train_loader.batch_size, shuffle=False,
                num_workers=getattr(train_loader, 'num_workers', 0) or 2,
                pin_memory=train_loader.pin_memory)
            print(f"Validation split: train {train_size} | val {val_size}")
        else:
            self.train_loader = train_loader
            self.val_loader = None

        self.test_loader = test_loader

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
        self.grad_clip_norm = grad_clip_norm

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=lr_factor,
            patience=lr_patience, min_lr=1e-6
        )

        self.patience = patience
        self.best_val_f1 = 0
        self.early_stop_counter = 0

        # Periodic evaluation stride: prefer periodic_eval_records (by record count, auto-adapts
        # to batch_size), otherwise use periodic_eval_steps (by batch count)
        if periodic_eval_records is not None:
            batch_size_val = train_loader.batch_size
            self.periodic_eval_steps = max(1, int(periodic_eval_records / batch_size_val))
            self._periodic_eval_mode = 'records'
            self._periodic_eval_records = periodic_eval_records
        else:
            self.periodic_eval_steps = periodic_eval_steps
            self._periodic_eval_mode = 'steps'
            self._periodic_eval_records = None

        # Train-set sampled-evaluation buffer: reused by _periodic_eval, keeps the most recent 50 batches
        self._train_eval_buffer = {'preds': [], 'labels': [], 'loss_sum': 0.0, 'count': 0}
        self._train_eval_buffer_size = 50

        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        self.global_train_step = 0  # cumulative batch step (used as the x-axis)

    def compute_metrics(self, predictions, labels, threshold=0.5):
        """Compute Precision, Recall, F1, AUC"""
        if len(predictions) == 0:
            return 0.0, 0.0, 0.0, 0.0

        binary_predictions = (predictions > threshold).float()
        labels_np = labels.cpu().numpy()
        predictions_np = binary_predictions.cpu().numpy()
        probs_np = predictions.cpu().numpy()

        try:
            precision = precision_score(labels_np, predictions_np, zero_division=0)
            recall = recall_score(labels_np, predictions_np, zero_division=0)
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            auc = roc_auc_score(labels_np, probs_np) if len(np.unique(labels_np)) > 1 else 0.5
        except:
            precision, recall, f1, auc = 0.0, 0.0, 0.0, 0.5

        return precision, recall, f1, auc

    def train_epoch(self):
        """Train one epoch and return cumulative metrics. Periodic evaluation is handled by the train() main loop."""
        self.model.train()
        total_bce_loss = 0.0
        total_loss_val = 0.0
        all_predictions = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc='Training')
        n_batches = len(self.train_loader)

        for batch_idx, (link_indices, current_times, labels) in enumerate(pbar):
            link_indices = link_indices.to(self.device)
            current_times = current_times.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(self.static_data, link_indices, current_times)
            if len(predictions) <= 1:
                continue

            loss = self.criterion(predictions, labels)
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_bce_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_labels.append(labels.detach())

            self.global_train_step += 1

            pbar.set_postfix({
                'BCE': f'{loss.item():.4f}',
                'Step': self.global_train_step,
            })

        if len(all_predictions) == 0:
            return 0.0, 0.0, 0.0, 0.0

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        avg_bce = total_bce_loss / max(n_batches, 1)
        precision, recall, f1, auc = self.compute_metrics(all_predictions, all_labels)
        return avg_bce, precision, recall, f1, auc

    def _evaluate_on_loader(self, loader, loader_name='Eval', return_raw=False):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for link_indices, current_times, labels in loader:
                link_indices = link_indices.to(self.device)
                current_times = current_times.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(self.static_data, link_indices, current_times)
                if len(predictions) <= 1:
                    continue
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()

                all_predictions.append(predictions)
                all_labels.append(labels)

        if len(all_predictions) == 0:
            if return_raw:
                return 0.0, 0.0, 0.0, 0.0, 0.0, torch.tensor([]), torch.tensor([])
            return 0.0, 0.0, 0.0, 0.0, 0.0

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        avg_loss = total_loss / max(len(loader), 1)
        precision, recall, f1, auc = self.compute_metrics(all_predictions, all_labels)

        if return_raw:
            return avg_loss, precision, recall, f1, auc, all_predictions, all_labels
        return avg_loss, precision, recall, f1, auc

    def compute_factset_quantile(self, step=None, max_factset_edges=None,
                                 test_scores=None, test_pos_scores=None, test_neg_scores=None):
        if not self.factset_edges or self.node_mapping is None:
            return None

        self.model.eval()

        factset_edges = self.factset_edges
        if max_factset_edges is not None and len(factset_edges) > max_factset_edges:
            import random
            factset_edges = random.sample(factset_edges, max_factset_edges)

        valid_pairs = []
        valid_years = []
        for src_id, tgt_id, year in tqdm(factset_edges, desc="Factset: preparing pairs", leave=False):
            if src_id in self.node_mapping and tgt_id in self.node_mapping:
                valid_pairs.append([self.node_mapping[src_id], self.node_mapping[tgt_id]])
                valid_years.append(year)

        if not valid_pairs:
            return None

        factset_scores = []
        batch_size = 128
        with torch.no_grad():
            for i in tqdm(range(0, len(valid_pairs), batch_size),
                          desc="Factset: scoring edges", leave=False):
                batch_pairs = torch.tensor(valid_pairs[i:i+batch_size],
                                           dtype=torch.long, device=self.device)
                batch_times = torch.tensor(valid_years[i:i+batch_size],
                                           dtype=torch.float, device=self.device)
                preds = self.model(self.static_data, batch_pairs, batch_times)
                if len(preds) > 0:
                    factset_scores.extend(preds.cpu().numpy().flatten().tolist())

        if not factset_scores:
            return None

        # Scores of all test-set edges (separating positive/negative examples); reuse if the caller passed precomputed results
        if test_scores is not None and test_pos_scores is not None and test_neg_scores is not None:
            pass
        else:
            test_scores = []
            test_pos_scores = []
            test_neg_scores = []
            with torch.no_grad():
                for link_indices, current_times, labels in tqdm(self.test_loader,
                        desc="Factset: scoring test set", leave=False):
                    link_indices = link_indices.to(self.device)
                    current_times = current_times.to(self.device)
                    predictions = self.model(self.static_data, link_indices, current_times)
                    if len(predictions) > 0:
                        preds_np = predictions.cpu().numpy().flatten()
                        test_scores.extend(preds_np.tolist())
                        labels_np = labels.cpu().numpy().flatten()
                        test_pos_scores.extend(preds_np[labels_np == 1].tolist())
                        test_neg_scores.extend(preds_np[labels_np == 0].tolist())

        if not test_scores:
            return None

        # TensorBoard histograms
        if self.use_tensorboard and step is not None:
            self.writer.add_histogram('Score_Distribution/TestSet_All', np.array(test_scores), step)
            self.writer.add_histogram('Score_Distribution/Factset', np.array(factset_scores), step)
            if test_pos_scores:
                self.writer.add_histogram('Score_Distribution/TestSet_Positive',
                                          np.array(test_pos_scores), step)
            if test_neg_scores:
                self.writer.add_histogram('Score_Distribution/TestSet_Negative',
                                          np.array(test_neg_scores), step)

        # Wasserstein-1 distance
        factset_arr = np.array(factset_scores)
        wass_pos = wasserstein_distance(factset_arr, np.array(test_pos_scores)) if test_pos_scores else None
        wass_neg = wasserstein_distance(factset_arr, np.array(test_neg_scores)) if test_neg_scores else None

        if self.use_tensorboard and step is not None:
            if wass_pos is not None:
                self.writer.add_scalar('Monitor/Wasserstein_Factset_vs_Pos', wass_pos, step)
            if wass_neg is not None:
                self.writer.add_scalar('Monitor/Wasserstein_Factset_vs_Neg', wass_neg, step)

        wass_diff = None
        if wass_pos is not None and wass_neg is not None:
            wass_diff = wass_neg - wass_pos
            if self.use_tensorboard and step is not None:
                self.writer.add_scalar('Monitor/Wasserstein_Diff', wass_diff, step)

        test_scores_sorted = np.sort(test_scores)
        percentiles = [np.searchsorted(test_scores_sorted, s) / len(test_scores_sorted)
                       for s in factset_scores]
        quantile = float(np.mean(percentiles))

        if self.use_tensorboard and step is not None:
            self.writer.add_scalar('Monitor/Factset_Quantile', quantile, step)

        result = {'quantile': quantile}
        if wass_pos is not None:
            result['wasserstein_pos'] = wass_pos
        if wass_neg is not None:
            result['wasserstein_neg'] = wass_neg
        if wass_diff is not None:
            result['wasserstein_diff'] = wass_diff
        return result

    def _periodic_eval(self, step):
        """Run a full evaluation every N batch steps; TensorBoard metrics use step as the x-axis."""
        self.model.eval()

        test_loss, test_prec, test_rec, test_f1, test_auc, test_preds, test_labels = \
            self._evaluate_on_loader(self.test_loader, 'Test', return_raw=True)

        # Separate positive/negative example scores (reused by compute_factset_quantile)
        if len(test_preds) > 0:
            preds_np = test_preds.cpu().numpy().flatten()
            labels_np = test_labels.cpu().numpy().flatten()
            test_scores_all = preds_np.tolist()
            test_pos = preds_np[labels_np == 1].tolist()
            test_neg = preds_np[labels_np == 0].tolist()
        else:
            test_scores_all = test_pos = test_neg = []

        # Train-set sampled evaluation: read from the buffer to avoid re-iterating train_loader
        buf = self._train_eval_buffer
        if buf['count'] > 0 and len(buf['preds']) > 0:
            tp = torch.cat(buf['preds'])
            tl = torch.cat(buf['labels'])
            train_bce = buf['loss_sum'] / buf['count']
            train_prec, train_rec, train_f1, train_auc = self.compute_metrics(tp, tl)
        else:
            train_bce = train_prec = train_rec = train_f1 = train_auc = 0.0

        # Factset evaluation reuses the test-set scores to avoid duplicate inference
        factset_result = self.compute_factset_quantile(
            step=step, max_factset_edges=self.max_factset_edges,
            test_scores=test_scores_all, test_pos_scores=test_pos,
            test_neg_scores=test_neg)

        if self.use_tensorboard:
            self.writer.add_scalar('Train/Epoch_BCE', train_bce, step)
            self.writer.add_scalar('Train/Epoch_Loss', train_bce, step)
            self.writer.add_scalar('Train/Epoch_F1', train_f1, step)
            self.writer.add_scalar('Train/Epoch_AUC', train_auc, step)
            self.writer.add_scalar('Test/Epoch_Loss', test_loss, step)
            self.writer.add_scalar('Test/Epoch_F1', test_f1, step)
            self.writer.add_scalar('Test/Epoch_AUC', test_auc, step)

        self.model.train()

        print(f'\n  [Step {step}] Periodic Evaluation')
        print(f'  Train - BCE: {train_bce:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}')
        print(f'  Test  - Loss: {test_loss:.4f}, Precision: {test_prec:.4f}, '
              f'Recall: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}')
        if factset_result is not None:
            fq = factset_result['quantile']
            print(f'  Factset Quantile: {fq:.4f} (higher is better)')
            if 'wasserstein_diff' in factset_result:
                wd = factset_result['wasserstein_diff']
                print(f'  Wasserstein Diff (Neg-Pos): {wd:.6f} (larger is better)')

        return test_f1, test_auc

    def save_test_predictions_npy(self, npy_path, checkpoint_path=None):
        """Run inference on the test set and save prediction scores as .npy (dict: 'test_predictions'/'neg_predictions'), for ROC/AUC curve plotting."""
        import numpy as np

        if checkpoint_path is not None:
            print(f"[SavePred] loading best model: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))

        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Saving test predictions')
            for link_indices, current_times, labels in pbar:
                link_indices = link_indices.to(self.device)
                current_times = current_times.to(self.device)
                labels = labels.to(self.device).float()
                predictions = self.model(self.static_data, link_indices, current_times)
                if len(predictions) > 0:
                    all_predictions.append(predictions.cpu().numpy().flatten())
                    all_labels.append(labels.cpu().numpy().flatten())

        if len(all_predictions) == 0:
            print("[SavePred] WARNING: no valid test-set predictions, skipping save")
            return

        all_preds = np.concatenate(all_predictions)
        all_labs = np.concatenate(all_labels)

        pos_scores = all_preds[all_labs == 1]
        neg_scores = all_preds[all_labs == 0]

        save_dict = {
            'test_predictions': [pos_scores],
            'neg_predictions': [neg_scores],
        }

        os.makedirs(os.path.dirname(npy_path) or '.', exist_ok=True)
        np.save(npy_path, save_dict)
        print(f"[SavePred] test-set predictions saved: {npy_path}")
        print(f"  positives: {len(pos_scores)}, negatives: {len(neg_scores)}")
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            from sklearn.metrics import roc_auc_score
            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
            y_scores = np.concatenate([pos_scores, neg_scores])
            try:
                auc = roc_auc_score(y_true, y_scores)
                print(f"  AUC: {auc:.4f}")
            except:
                pass

    def train(self, num_epochs=2, save_path='best_model.pth'):
        print("SEAL Step-Level Training")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        if self._periodic_eval_mode == 'records':
            print(f"Epochs: {num_epochs} | Periodic Eval every: ~{self._periodic_eval_records} records "
                  f"(≈{self.periodic_eval_steps} steps @ batch_size={self.train_loader.batch_size})")
        else:
            print(f"Epochs: {num_epochs} | Periodic Eval every: {self.periodic_eval_steps} steps")
        print(f"Hyperparameters: lr={self.optimizer.param_groups[0]['lr']}, "
              f"weight_decay={self.optimizer.param_groups[0]['weight_decay']}, "
              f"grad_clip={self.grad_clip_norm}")

        best_f1 = 0
        best_step = 0
        best_auc = 0.0

        try:
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                self.model.train()

                pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
                epoch_bce = 0.0
                epoch_preds = []
                epoch_labels = []

                for batch_idx, (link_indices, current_times, labels) in enumerate(pbar):
                    link_indices = link_indices.to(self.device)
                    current_times = current_times.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()
                    predictions = self.model(self.static_data, link_indices, current_times)
                    if len(predictions) <= 1:
                        continue

                    loss = self.criterion(predictions, labels)
                    loss.backward()
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                    self.global_train_step += 1
                    epoch_bce += loss.item()
                    epoch_preds.append(predictions.detach())
                    epoch_labels.append(labels.detach())

                    # Update the train sampled-evaluation buffer, reused by _periodic_eval
                    buf = self._train_eval_buffer
                    buf['preds'].append(predictions.detach())
                    buf['labels'].append(labels.detach())
                    buf['loss_sum'] += loss.item()
                    buf['count'] += 1
                    if len(buf['preds']) > self._train_eval_buffer_size:
                        buf['preds'].pop(0)
                        buf['labels'].pop(0)

                    pbar.set_postfix({
                        'BCE': f'{loss.item():.4f}',
                        'Step': self.global_train_step,
                    })

                    if self.global_train_step % self.periodic_eval_steps == 0:
                        test_f1, test_auc = self._periodic_eval(self.global_train_step)

                        # Save best model (based on step-level F1)
                        if test_f1 > best_f1:
                            best_f1 = test_f1
                            best_step = self.global_train_step
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            torch.save({
                                'step': self.global_train_step,
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'test_f1': test_f1,
                                'test_auc': test_auc,
                            }, save_path)
                            print(f'  >> Saved best model (Step {best_step}, '
                                  f'Test F1={best_f1:.4f}, AUC={test_auc:.4f})')
                            # Also save test-set predictions (overwrite, consistent with best_model.pth)
                            best_npy_path = os.path.join(os.path.dirname(save_path), 'model_predictions_best.npy')
                            self.save_test_predictions_npy(best_npy_path, checkpoint_path=None)
                            self.model.train()

                        if test_auc > best_auc:
                            best_auc = test_auc
                            auc_save_path = os.path.join(os.path.dirname(save_path), 'best_auc_model.pth')
                            torch.save({
                                'step': self.global_train_step,
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'test_f1': test_f1,
                                'test_auc': test_auc,
                            }, auc_save_path)
                            print(f'  >> Saved best-AUC model (Step {self.global_train_step}, '
                                  f'AUC={test_auc:.4f})')
                            # Also save test-set predictions (overwrite, consistent with best_auc_model.pth)
                            best_auc_npy_path = os.path.join(os.path.dirname(save_path), 'model_predictions_best_auc.npy')
                            self.save_test_predictions_npy(best_auc_npy_path, checkpoint_path=None)
                            self.model.train()

                # Epoch summary
                if epoch_preds:
                    all_preds = torch.cat(epoch_preds)
                    all_labs = torch.cat(epoch_labels)
                    avg_bce = epoch_bce / max(len(self.train_loader), 1)
                    ep_prec, ep_rec, ep_f1, ep_auc = self.compute_metrics(all_preds, all_labs)
                    epoch_duration = time.time() - epoch_start_time
                    if self.use_tensorboard:
                        self.writer.add_scalar('Time/Epoch_Duration', epoch_duration, epoch)
                    print(f'Epoch {epoch+1} - BCE: {avg_bce:.4f}, '
                          f'F1: {ep_f1:.4f}, AUC: {ep_auc:.4f} '
                          f'({epoch_duration:.1f}s)')

            # Training finished, load the best model
            print(f"\nLoading best model (Step {best_step}, F1={best_f1:.4f})...")
            checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            test_loss, test_prec, test_rec, test_f1, test_auc = \
                self._evaluate_on_loader(self.test_loader, 'Final Test')
            print(f"Final test: Loss={test_loss:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

        finally:
            if self.use_tensorboard:
                self.writer.close()

        print(f"\nTraining complete! Best Step: {best_step}, F1: {best_f1:.4f}")


if __name__ == "__main__":
    import argparse
    from Training.config_loader import load_config
    from utils import import_attr, resolve_auto_kwargs

    parser = argparse.ArgumentParser(description='Temp-SEAL Step-Level Training')
    parser.add_argument('--config', '-c', default='seal',
                        help='model config name (default: seal)')
    parser.add_argument('--device', '-d', default='auto',
                        help='device (auto/cuda/cpu)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device

    cfg = load_config(args.config)
    ds_cfg = cfg['dataset']
    model_cfg = cfg['model']
    trainer_cfg = cfg['trainer']

    create_dataloaders_fn = import_attr(ds_cfg['module'], 'create_dataloaders')

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results',
                            cfg.get('output_subdir', 'seal'))
    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime as dt_seal
    run_name = dt_seal.now().strftime("%m%d-%H%M")

    static_data, train_loader, _val_loader, test_loader, full_dataset = create_dataloaders_fn(
        negative_ratio=ds_cfg.get('negative_ratio', 1),
        batch_size=ds_cfg['batch_size'],
        train_ratio=ds_cfg.get('train_ratio', 0.8),
        toy_mode=cfg.get('toy_mode', False),
        filter_factset_neg=ds_cfg.get('filter_factset_neg', False),
        intra_industry_neg=ds_cfg.get('intra_industry_neg', True),
        use_pred_neg=ds_cfg.get('use_pred_neg', True),
    )

    factset_edges = getattr(full_dataset, 'factset_edges', [])
    node_mapping = getattr(full_dataset, 'node_mapping', None)
    reverse_node_mapping = getattr(full_dataset, 'reverse_node_mapping', None)

    auto_context = {
        'num_features': static_data.x.size(1),
        'num_nodes': static_data.num_nodes,
        'time_steps': sorted(static_data.edge_time.unique().tolist()),
        'hidden_dims': static_data.x.size(1),
        'device': device,
        'batch_size': ds_cfg['batch_size'],
    }
    model_kwargs = resolve_auto_kwargs(model_cfg['kwargs'], auto_context)
    ModelClass = import_attr(model_cfg['module'], model_cfg['class'])
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

    trainer_kw = trainer_cfg.get('kwargs', {})
    trainer = SEALTrainer(
        model=model,
        static_data=static_data,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        use_tensorboard=trainer_kw.get('use_tensorboard', True),
        log_dir=os.path.join(save_dir, run_name),
        lr=trainer_kw.get('lr', 0.001),
        weight_decay=trainer_kw.get('weight_decay', 1e-4),
        grad_clip_norm=trainer_kw.get('grad_clip_norm', 1.0),
        val_ratio=trainer_kw.get('val_ratio', 0.1),
        patience=trainer_kw.get('patience', 15),
        lr_patience=trainer_kw.get('lr_patience', 8),
        lr_factor=trainer_kw.get('lr_factor', 0.5),
        factset_edges=factset_edges,
        node_mapping=node_mapping,
        reverse_node_mapping=reverse_node_mapping,
        max_factset_edges=trainer_cfg.get('max_factset_edges'),
        periodic_eval_steps=trainer_kw.get('periodic_eval_steps', 1000),
        periodic_eval_records=trainer_kw.get('periodic_eval_records', None),
    )

    trainer.train(
        num_epochs=trainer_cfg.get('num_epochs', 2),
        save_path=os.path.join(save_dir, run_name, 'best_model.pth'),
    )

