"""
SEAL trainer (step-level metrics version): a full evaluation every N batches.
Monitoring metrics (Loss/F1/AUC/BCE/Factset quantile/Wasserstein) are logged per batch step,
with metric naming aligned to trainer_common.py, to ease dual-axis display by step.

Evaluation protocol: the validation split drives checkpoint selection and the test split is
reported only. When no validation split is available, the test split takes over the selection role,
which reproduces the legacy behaviour.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
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
                 val_loader=None,
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

        # FactSet edge scores, computed once per evaluation step and shared by the two scopes
        self._factset_cache_step = None
        self._factset_cache = (None, [])
        self.epoch_durations = []

        # Validation split. Priority order:
        #   1. an external val_loader (the year-stratified validation split built by
        #      create_dataloaders): the same definition the other backbones use, so the reported
        #      numbers stay comparable, and the split is disjoint from the training pool, so no
        #      training data is spent on it;
        #   2. otherwise the legacy random_split of the training pool (val_ratio > 0);
        #   3. otherwise no validation split at all, and selection falls back to the test split.
        if val_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
            print(f"Validation split (external): train {len(train_loader.dataset)} | "
                  f"val {len(val_loader.dataset)}")
        elif val_ratio > 0:
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
            print(f"Validation split (random_split): train {train_size} | val {val_size}")
        else:
            self.train_loader = train_loader
            self.val_loader = None
            print("Validation split: none (checkpoint selection falls back to the test split)")

        self.test_loader = test_loader
        self.selection_split = 'val' if self.val_loader is not None else 'test'

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
        self.grad_clip_norm = grad_clip_norm

        # LR schedule: stepped once per periodic evaluation on the selection metric (see train()),
        # so lr_patience counts EVALUATIONS rather than epochs.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=lr_factor,
            patience=lr_patience, min_lr=1e-6
        )
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.scheduler_evals = 0  # number of scheduler.step() calls so far
        self.lr_events = []       # [{epoch, step, eval_index, f1, lr_before, lr_after}, ...]

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

    def compute_factset_quantile(self, step=None, max_factset_edges=None, scope=None,
                                 split_data=None):
        """FactSet statistics of one data split: the mean quantile of the FactSet edges within that
        split's scores, and the Wasserstein-1 distance between the FactSet score distribution and the
        split's positive/negative examples.

        Args:
            scope: 'val' or 'test'. Defaults to the split used for model selection.
            split_data: optional {scope: (all_scores, pos_scores, neg_scores)} of precomputed scores,
                so the statistics reuse an evaluation pass instead of re-running a loader.
        Returns:
            dict {'quantile', 'wasserstein_pos', 'wasserstein_neg', 'wasserstein_diff', 'scope',
            'test'} or None; 'test' carries the test-split statistics whenever the selection split is
            the validation split, so monitoring and reporting never share a curve.
        """
        if scope is None:
            scope = self.selection_split

        result = self._factset_stats(scope, step=step, max_factset_edges=max_factset_edges,
                                     split_data=split_data)
        if result is None:
            return None

        result['test'] = None
        if scope != 'test':
            result['test'] = self._factset_stats('test', step=step,
                                                 max_factset_edges=max_factset_edges,
                                                 split_data=split_data)
        return result

    def _factset_stats(self, scope, step=None, max_factset_edges=None, split_data=None):
        """FactSet quantile / Wasserstein statistics of a single split (scope: 'val' or 'test')."""
        if not self.factset_edges or self.node_mapping is None:
            return None

        self.model.eval()

        factset_scores = self._factset_edge_scores(step=step, max_factset_edges=max_factset_edges)
        if not factset_scores:
            return None

        if split_data and scope in split_data:
            split_scores, pos_scores, neg_scores = split_data[scope]
        else:
            loader = self.val_loader if scope == 'val' else self.test_loader
            if loader is None:
                return None
            split_scores, pos_scores, neg_scores = self._score_loader_splits(loader, scope)

        if not split_scores:
            return None

        split_label = 'ValSet' if scope == 'val' else 'TestSet'
        # Monitor/* always describes the split that drives model selection; the test split gets its
        # own MonitorTest/* prefix whenever the two splits differ.
        monitor = 'Monitor' if scope == self.selection_split else 'MonitorTest'

        factset_arr = np.array(factset_scores)
        if self.use_tensorboard and step is not None:
            self.writer.add_histogram('Score_Distribution/Factset', factset_arr, step)
            self.writer.add_histogram(f'Score_Distribution/{split_label}_All',
                                      np.array(split_scores), step)
            if pos_scores:
                self.writer.add_histogram(f'Score_Distribution/{split_label}_Positive',
                                          np.array(pos_scores), step)
            if neg_scores:
                self.writer.add_histogram(f'Score_Distribution/{split_label}_Negative',
                                          np.array(neg_scores), step)

        # Wasserstein-1 distance: FactSet scores vs the split's positive/negative scores
        wass_pos = wasserstein_distance(factset_arr, np.array(pos_scores)) if pos_scores else None
        wass_neg = wasserstein_distance(factset_arr, np.array(neg_scores)) if neg_scores else None
        wass_diff = wass_neg - wass_pos if (wass_pos is not None and wass_neg is not None) else None

        if self.use_tensorboard and step is not None:
            if wass_pos is not None:
                self.writer.add_scalar(f'{monitor}/Wasserstein_Factset_vs_Pos', wass_pos, step)
            if wass_neg is not None:
                self.writer.add_scalar(f'{monitor}/Wasserstein_Factset_vs_Neg', wass_neg, step)
            if wass_diff is not None:
                self.writer.add_scalar(f'{monitor}/Wasserstein_Diff', wass_diff, step)

        # Quantile of each FactSet edge within the split's score distribution
        scores_sorted = np.sort(split_scores)
        percentiles = [np.searchsorted(scores_sorted, s) / len(scores_sorted)
                       for s in factset_scores]
        quantile = float(np.mean(percentiles))

        if self.use_tensorboard and step is not None:
            self.writer.add_scalar(f'{monitor}/Factset_Quantile', quantile, step)

        result = {'quantile': quantile, 'scope': scope}
        if wass_pos is not None:
            result['wasserstein_pos'] = wass_pos
        if wass_neg is not None:
            result['wasserstein_neg'] = wass_neg
        if wass_diff is not None:
            result['wasserstein_diff'] = wass_diff
        return result

    def _factset_edge_scores(self, step=None, max_factset_edges=None):
        """Score the FactSet edges, cached per step so both scopes share one inference pass."""
        cacheable = step is not None
        if cacheable and self._factset_cache_step == step \
                and self._factset_cache[0] == max_factset_edges:
            return self._factset_cache[1]

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

        factset_scores = []
        if valid_pairs:
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

        if cacheable:
            self._factset_cache_step = step
            self._factset_cache = (max_factset_edges, factset_scores)
        return factset_scores

    def _score_loader_splits(self, loader, scope):
        """All / positive / negative scores of one split."""
        scores_all, pos_scores, neg_scores = [], [], []
        with torch.no_grad():
            for link_indices, current_times, labels in tqdm(
                    loader, desc=f"Factset: scoring {scope} set", leave=False):
                link_indices = link_indices.to(self.device)
                current_times = current_times.to(self.device)
                predictions = self.model(self.static_data, link_indices, current_times)
                if len(predictions) > 0:
                    preds_np = predictions.cpu().numpy().flatten()
                    labels_np = labels.cpu().numpy().flatten()
                    scores_all.extend(preds_np.tolist())
                    pos_scores.extend(preds_np[labels_np == 1].tolist())
                    neg_scores.extend(preds_np[labels_np == 0].tolist())
        return scores_all, pos_scores, neg_scores

    @staticmethod
    def _split_sample_scores(preds, labels):
        """(all, positive, negative) score lists from one raw evaluation pass."""
        if preds is None or len(preds) == 0:
            return [], [], []
        preds_np = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        return (preds_np.tolist(),
                preds_np[labels_np == 1].tolist(),
                preds_np[labels_np == 0].tolist())

    @staticmethod
    def _fmt_factset_report(result, split_label):
        """Console lines for one split's FactSet statistics."""
        if result is None:
            return []
        lines = [f'  Factset Quantile ({split_label}): {result["quantile"]:.4f} '
                 f'(higher is better)']
        if 'wasserstein_diff' in result:
            lines.append(f'  Wasserstein Diff ({split_label}, Neg-Pos): '
                         f'{result["wasserstein_diff"]:.6f} (larger is better)')
        return lines

    def _checkpoint_dict(self, epoch, select_metrics, test_metrics):
        """Checkpoint payload: the weights plus every metric reported in the paper."""
        factset = select_metrics.get('factset')
        factset_test = test_metrics.get('factset')
        ckpt = {
            'step': self.global_train_step,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'selection_split': self.selection_split,
            # Selection split (drives checkpoint selection)
            'train_bce': select_metrics['train_bce'],
            'val_loss': select_metrics['loss'],
            'val_f1': select_metrics['f1'],
            'val_auc': select_metrics['auc'],
            # Test split (reported only)
            'test_loss': test_metrics['loss'],
            'test_f1': test_metrics['f1'],
            'test_auc': test_metrics['auc'],
        }
        for key in ('quantile', 'wasserstein_pos', 'wasserstein_neg', 'wasserstein_diff'):
            if factset is not None and key in factset:
                ckpt['factset_' + key] = factset[key]
            if factset_test is not None and key in factset_test:
                ckpt['factset_' + key + '_test'] = factset_test[key]
        ckpt['factset_scope'] = self.selection_split
        return ckpt

    def _periodic_eval(self, step):
        """Run a full evaluation every N batch steps; TensorBoard metrics use step as the x-axis.

        The validation split drives checkpoint selection, the test split is reported only.
        Returns (selection_metrics, test_metrics) dicts.
        """
        self.model.eval()

        select_is_val = self.val_loader is not None
        select_loader = self.val_loader if select_is_val else self.test_loader
        select_label = 'Val' if select_is_val else 'Test'

        s_loss, s_prec, s_rec, s_f1, s_auc, s_preds, s_labels = \
            self._evaluate_on_loader(select_loader, select_label, return_raw=True)
        s_scores, s_pos, s_neg = self._split_sample_scores(s_preds, s_labels)

        if select_is_val:
            t_loss, t_prec, t_rec, t_f1, t_auc, t_preds, t_labels = \
                self._evaluate_on_loader(self.test_loader, 'Test', return_raw=True)
            t_scores, t_pos, t_neg = self._split_sample_scores(t_preds, t_labels)
        else:
            # No validation split: the test split takes over the selection role
            t_loss, t_prec, t_rec, t_f1, t_auc = s_loss, s_prec, s_rec, s_f1, s_auc
            t_scores, t_pos, t_neg = s_scores, s_pos, s_neg

        # Train-set sampled evaluation: read from the buffer to avoid re-iterating train_loader
        buf = self._train_eval_buffer
        if buf['count'] > 0 and len(buf['preds']) > 0:
            tp = torch.cat(buf['preds'])
            tl = torch.cat(buf['labels'])
            train_bce = buf['loss_sum'] / buf['count']
            train_prec, train_rec, train_f1, train_auc = self.compute_metrics(tp, tl)
        else:
            train_bce = train_prec = train_rec = train_f1 = train_auc = 0.0

        # FactSet statistics: selection split (drives selection) + test split (reported). The split
        # scores come from the evaluation passes above and the FactSet edges are scored once per
        # step, so no inference is duplicated.
        split_data = {select_label.lower(): (s_scores, s_pos, s_neg)}
        if select_is_val:
            split_data['test'] = (t_scores, t_pos, t_neg)
        select_factset = self.compute_factset_quantile(
            step=step, max_factset_edges=self.max_factset_edges,
            scope=select_label.lower(), split_data=split_data)
        test_factset = select_factset.get('test') if (select_factset is not None
                                                      and select_is_val) else select_factset

        if self.use_tensorboard:
            self.writer.add_scalar('Train/Epoch_BCE', train_bce, step)
            self.writer.add_scalar('Train/Epoch_Loss', train_bce, step)
            self.writer.add_scalar('Train/Epoch_F1', train_f1, step)
            self.writer.add_scalar('Train/Epoch_AUC', train_auc, step)
            self.writer.add_scalar(f'{select_label}/Epoch_Loss', s_loss, step)
            self.writer.add_scalar(f'{select_label}/Epoch_F1', s_f1, step)
            self.writer.add_scalar(f'{select_label}/Epoch_AUC', s_auc, step)
            if select_is_val:
                self.writer.add_scalar('Test/Epoch_Loss', t_loss, step)
                self.writer.add_scalar('Test/Epoch_F1', t_f1, step)
                self.writer.add_scalar('Test/Epoch_AUC', t_auc, step)

        self.model.train()

        print(f'\n  [Step {step}] Periodic Evaluation '
              f'(lr={self.optimizer.param_groups[0]["lr"]:.2e})')
        print(f'  Train - BCE: {train_bce:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}')
        print(f'  {select_label:<5} - Loss: {s_loss:.4f}, Precision: {s_prec:.4f}, '
              f'Recall: {s_rec:.4f}, F1: {s_f1:.4f}, AUC: {s_auc:.4f}')
        if select_is_val:
            print(f'  Test  - Loss: {t_loss:.4f}, Precision: {t_prec:.4f}, '
                  f'Recall: {t_rec:.4f}, F1: {t_f1:.4f}, AUC: {t_auc:.4f}')
        for label, res in (('Sel', select_factset), ('Test', test_factset)):
            for line in self._fmt_factset_report(res, label):
                print(line)

        select_metrics = {
            'loss': s_loss, 'prec': s_prec, 'rec': s_rec, 'f1': s_f1, 'auc': s_auc,
            'train_bce': train_bce, 'factset': select_factset,
        }
        test_metrics = {
            'loss': t_loss, 'prec': t_prec, 'rec': t_rec, 'f1': t_f1, 'auc': t_auc,
            'factset': test_factset,
        }
        return select_metrics, test_metrics

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
        print(f"LR schedule: ReduceLROnPlateau(mode=max on {self.selection_split} F1, "
              f"factor={self.lr_factor}, patience={self.lr_patience} evaluations, min_lr=1e-6)")
        # Guard: the schedule counts evaluations, so report up-front when it cannot trigger at all
        # instead of leaving lr_patience a silent no-op.
        total_evals = (max(len(self.train_loader), 1) * num_epochs) // self.periodic_eval_steps
        if total_evals <= self.lr_patience:
            print(f"  [warn] LR schedule will NOT trigger in this run: ~{total_evals} periodic "
                  f"evaluations <= lr_patience={self.lr_patience} "
                  f"(lower lr_patience or raise num_epochs)")

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
                        select_metrics, test_metrics = self._periodic_eval(self.global_train_step)
                        # Selection uses the selection split (validation when configured, otherwise
                        # the test split); the test split is only reported.
                        select_f1, select_auc = select_metrics['f1'], select_metrics['auc']

                        # LR schedule: stepped once per periodic evaluation on the same metric that
                        # drives checkpoint selection, halving the LR after lr_patience consecutive
                        # evaluations without improvement (floored at min_lr=1e-6).
                        lr_before = self.optimizer.param_groups[0]['lr']
                        self.scheduler.step(select_f1)
                        self.scheduler_evals += 1
                        lr_after = self.optimizer.param_groups[0]['lr']
                        if lr_after < lr_before:
                            self.lr_events.append({
                                'epoch': epoch, 'step': self.global_train_step,
                                'eval_index': self.scheduler_evals, 'f1': select_f1,
                                'lr_before': lr_before, 'lr_after': lr_after,
                            })
                            print(f'  >> LR reduced ({self.selection_split} F1={select_f1:.4f} did not '
                                  f'improve for {self.lr_patience} evaluations): '
                                  f'{lr_before:.2e} -> {lr_after:.2e}')

                        # Save best model (based on the step-level F1 of the selection split)
                        if select_f1 > best_f1:
                            best_f1 = select_f1
                            best_step = self.global_train_step
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            torch.save(self._checkpoint_dict(epoch, select_metrics, test_metrics),
                                       save_path)
                            print(f'  >> Saved best model (Step {best_step}, '
                                  f'{self.selection_split} F1={best_f1:.4f}, '
                                  f'Test F1={test_metrics["f1"]:.4f}, '
                                  f'Test AUC={test_metrics["auc"]:.4f})')
                            # Also save test-set predictions (overwrite, consistent with best_model.pth)
                            best_npy_path = os.path.join(os.path.dirname(save_path), 'model_predictions_best.npy')
                            self.save_test_predictions_npy(best_npy_path, checkpoint_path=None)
                            self.model.train()

                        if select_auc > best_auc:
                            best_auc = select_auc
                            auc_save_path = os.path.join(os.path.dirname(save_path), 'best_auc_model.pth')
                            torch.save(self._checkpoint_dict(epoch, select_metrics, test_metrics),
                                       auc_save_path)
                            print(f'  >> Saved best-AUC model (Step {self.global_train_step}, '
                                  f'{self.selection_split} AUC={select_auc:.4f})')
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
                    self.epoch_durations.append(epoch_duration)
                    if self.use_tensorboard:
                        self.writer.add_scalar('Time/Epoch_Duration', epoch_duration, epoch)
                    print(f'Epoch {epoch+1} - BCE: {avg_bce:.4f}, '
                          f'F1: {ep_f1:.4f}, AUC: {ep_auc:.4f} '
                          f'({epoch_duration:.1f}s)')

            # Training finished, load the best model
            print(f"\nLoading best model (Step {best_step}, "
                  f"{self.selection_split} F1={best_f1:.4f})...")
            checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.val_loader is not None:
                val_loss, val_prec, val_rec, val_f1, val_auc = \
                    self._evaluate_on_loader(self.val_loader, 'Final Val')
                print(f"Final validation: Loss={val_loss:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
            test_loss, test_prec, test_rec, test_f1, test_auc = \
                self._evaluate_on_loader(self.test_loader, 'Final Test')
            print(f"Final test: Loss={test_loss:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

            avg_epoch_seconds = float(np.mean(self.epoch_durations)) if self.epoch_durations else None
            # Run summary: every paper-facing metric of the selected checkpoint in one JSON,
            # mirroring summary.yaml for the models trained with trainer_common.py.
            summary = {key: checkpoint.get(key) for key in (
                'step', 'epoch', 'selection_split', 'train_bce',
                'val_loss', 'val_f1', 'val_auc',
                'test_loss', 'test_f1', 'test_auc',
                'factset_quantile', 'wasserstein_pos', 'wasserstein_neg', 'wasserstein_diff',
                'factset_quantile_test', 'wasserstein_diff_test')}
            summary['epochs'] = num_epochs
            summary['best_auc'] = best_auc
            summary['final_lr'] = self.optimizer.param_groups[0]['lr']
            summary['scheduler_evals'] = self.scheduler_evals
            summary['lr_events'] = self.lr_events
            summary['avg_epoch_seconds'] = avg_epoch_seconds
            summary['epoch_seconds'] = self.epoch_durations
            summary_path = os.path.join(os.path.dirname(save_path), 'run_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Run summary written: {summary_path}")

        finally:
            if self.use_tensorboard:
                self.writer.close()

        print(f"\nTraining complete! Best Step: {best_step}, "
              f"{self.selection_split} F1: {best_f1:.4f}")
        if self.epoch_durations:
            print(f"Average epoch time: {float(np.mean(self.epoch_durations)):.1f}s "
                  f"over {len(self.epoch_durations)} epochs")


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
    dataset_feature_kwargs = import_attr(ds_cfg['module'], 'dataset_feature_kwargs')

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results',
                            cfg.get('output_subdir', 'seal'))
    os.makedirs(save_dir, exist_ok=True)
    from datetime import datetime as dt_seal
    run_name = dt_seal.now().strftime("%m%d-%H%M")

    # create_dataloaders returns a 5-tuple (static_data, train_loader, val_loader, test_loader,
    # full_dataset). The validation split drives checkpoint selection and the test split is only
    # reported (see SEALTrainer.__init__), so it is passed on instead of being discarded.
    static_data, train_loader, val_loader, test_loader, full_dataset = create_dataloaders_fn(
        negative_ratio=ds_cfg.get('negative_ratio', 1),
        batch_size=ds_cfg['batch_size'],
        train_ratio=ds_cfg.get('train_ratio', 0.8),
        toy_mode=cfg.get('toy_mode', False),
        filter_factset_neg=ds_cfg.get('filter_factset_neg', False),
        intra_industry_neg=ds_cfg.get('intra_industry_neg', True),
        use_pred_neg=ds_cfg.get('use_pred_neg', True),
        **dataset_feature_kwargs(ds_cfg),
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
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        use_tensorboard=trainer_kw.get('use_tensorboard', True),
        log_dir=os.path.join(save_dir, run_name),
        lr=trainer_kw.get('lr', 0.001),
        weight_decay=trainer_kw.get('weight_decay', 1e-4),
        grad_clip_norm=trainer_kw.get('grad_clip_norm', 1.0),
        # val_ratio only applies when no external val_loader is supplied
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

