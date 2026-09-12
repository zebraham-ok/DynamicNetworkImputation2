"""
General-purpose trainer: supports BCE + Margin Ranking Loss (pairwise ranking loss),
Factset quantile monitoring, and AUC.

Model selection follows the standard protocol: the validation split drives checkpoint selection,
the composite score (0.5 * FactSet quantile + 0.5 * validation AUC) and early stopping, while the
test split is reported only. When no validation split is passed, the test split takes over that
role, which reproduces the legacy behaviour.

Usage:
    trainer = DynamicGraphTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        factset_edges=factset_edges,  # optional: [(src, tgt, year), ...] for the factset quantile metric
        margin_lambda=0.1,            # Margin Ranking Loss weight
        static_data=static_data,      # only for backbones whose forward takes the static graph (Temp-SEAL)
    )
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import wasserstein_distance
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


def _forward_takes_static_graph(model) -> bool:
    """Whether model.forward() expects the static graph as its first argument.

    Temp-SEAL extracts the k-hop enclosing subgraphs itself, so its signature is
    forward(data, link_indices, current_times); every other backbone is
    forward(link_indices, current_times). Detecting the signature here keeps a single
    inference entry point (DynamicGraphTrainer._predict) for all backbones, so Temp-SEAL is
    driven by exactly the same training/selection protocol as the other models.
    """
    try:
        params = list(inspect.signature(model.forward).parameters.values())
    except (TypeError, ValueError):
        return False
    if not params:
        return False
    first = params[0]
    return (first.name == 'data'
            and first.kind in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD))


class DynamicGraphTrainer:
    def __init__(self, model, train_loader, test_loader, val_loader=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 log_dir=None, use_tensorboard=True,
                 margin_lambda=0.1, margin=1.0, factset_edges=None,
                 node_mapping=None, reverse_node_mapping=None,
                 static_data=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Temp-SEAL builds its enclosing subgraphs from the static graph, so it needs one extra
        # argument in forward(). The graph is kept here (instead of being carried by the loaders)
        # because it is constant over the whole run.
        self.static_data = static_data.to(device) if static_data is not None else None
        self.forward_takes_static_graph = _forward_takes_static_graph(self.model)
        if self.forward_takes_static_graph and self.static_data is None:
            raise ValueError(
                f"{type(model).__name__}.forward() expects the static graph "
                "(forward(data, link_indices, current_times)) but no static_data was passed to "
                "DynamicGraphTrainer; pass static_data=<the first element returned by "
                "create_dataloaders>."
            )

        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.bce_criterion = nn.BCELoss()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.margin_lambda = margin_lambda

        # Factset external validation
        self.factset_edges = factset_edges if factset_edges else []
        self.node_mapping = node_mapping
        self.reverse_node_mapping = reverse_node_mapping

        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        self.global_train_step = 0
        self.global_val_step = 0
        self.global_test_step = 0

        # Model selection follows the standard protocol: the validation split drives checkpoint
        # selection and early stopping, while the test split is reported only. Without a
        # validation split we fall back to the test split, which reproduces the legacy behaviour.
        self.selection_split = 'val' if val_loader is not None else 'test'

        # Per-split (predictions, labels) of the last evaluate() call, so the FactSet statistics
        # reuse that pass instead of running an extra inference pass over the same split.
        self._eval_scores = {}
        # FactSet edge scores, computed once per epoch and shared by the scopes of that epoch.
        self._factset_cache_epoch = None
        self._factset_cache = (None, [])

        # Full per-epoch metric record, so callers can report the whole protocol without
        # re-deriving anything from the TensorBoard event files.
        self.history = []
        self.epoch_durations = []

    def _predict(self, link_indices, current_times):
        """Single inference entry point: hides the backbone-specific forward signature."""
        if self.forward_takes_static_graph:
            return self.model(self.static_data, link_indices, current_times)
        return self.model(link_indices, current_times)

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

    def compute_factset_quantile(self, epoch=None, max_factset_edges=None, scope=None):
        """
        Compute the FactSet statistics of one data split: the mean quantile of the FactSet edges
        within that split's scores, and the Wasserstein-1 distance between the FactSet score
        distribution and the split's positive/negative examples. A higher quantile and a smaller
        Wasserstein distance mean the model better captures true supply relationships.

        Args:
            scope: 'val' or 'test'. Defaults to the split used for model selection (the
                validation split when one is configured, the test split otherwise).
        Returns:
            dict {'quantile', 'wasserstein_pos', 'wasserstein_neg', 'wasserstein_diff', 'scope',
            'test'} or None. 'test' holds the test-split statistics whenever the selection split
            is the validation split, so that monitoring (validation, drives early stopping) and
            reporting (test) never share a curve.
        """
        if scope is None:
            scope = self.selection_split

        loader = self.test_loader if scope == 'test' else self.val_loader
        if loader is None:
            return None

        result = self._factset_stats(loader, scope, epoch=epoch,
                                     max_factset_edges=max_factset_edges)
        if result is None:
            return None

        result['test'] = None
        if scope != 'test':
            result['test'] = self._factset_stats(self.test_loader, 'test', epoch=epoch,
                                                 max_factset_edges=max_factset_edges)
        return result

    def _factset_stats(self, loader, scope, epoch=None, max_factset_edges=None):
        """FactSet quantile / Wasserstein statistics of a single split (scope: 'val' or 'test')."""
        if not self.factset_edges or self.node_mapping is None:
            return None

        self.model.eval()

        factset_scores = self._factset_edge_scores(epoch=epoch, max_factset_edges=max_factset_edges)
        if not factset_scores:
            return None

        scores_all, pos_scores, neg_scores = self._split_scores(loader, scope)
        if not scores_all:
            return None

        split_label = 'ValSet' if scope == 'val' else 'TestSet'
        # Monitor/* always describes the split that drives model selection; the test split gets its
        # own MonitorTest/* prefix whenever the two splits differ.
        monitor = 'Monitor' if scope == self.selection_split else 'MonitorTest'

        factset_arr = np.array(factset_scores)
        if self.use_tensorboard and epoch is not None:
            self.writer.add_histogram('Score_Distribution/Factset', factset_arr, epoch)
            self.writer.add_histogram(f'Score_Distribution/{split_label}_All', np.array(scores_all), epoch)
            if pos_scores:
                self.writer.add_histogram(f'Score_Distribution/{split_label}_Positive',
                                          np.array(pos_scores), epoch)
            if neg_scores:
                self.writer.add_histogram(f'Score_Distribution/{split_label}_Negative',
                                          np.array(neg_scores), epoch)

        # Wasserstein-1 distance: FactSet score distribution vs the split's positive/negative scores
        wass_pos = wasserstein_distance(factset_arr, np.array(pos_scores)) if pos_scores else None
        wass_neg = wasserstein_distance(factset_arr, np.array(neg_scores)) if neg_scores else None
        wass_diff = wass_neg - wass_pos if (wass_pos is not None and wass_neg is not None) else None

        if self.use_tensorboard and epoch is not None:
            if wass_pos is not None:
                self.writer.add_scalar(f'{monitor}/Wasserstein_Factset_vs_Pos', wass_pos, epoch)
            if wass_neg is not None:
                self.writer.add_scalar(f'{monitor}/Wasserstein_Factset_vs_Neg', wass_neg, epoch)
            if wass_diff is not None:
                self.writer.add_scalar(f'{monitor}/Wasserstein_Diff', wass_diff, epoch)

        # Quantile of each FactSet edge within the split's score distribution
        scores_sorted = np.sort(scores_all)
        percentiles = [np.searchsorted(scores_sorted, s) / len(scores_sorted)
                       for s in factset_scores]
        quantile = float(np.mean(percentiles))

        if self.use_tensorboard and epoch is not None:
            self.writer.add_scalar(f'{monitor}/Factset_Quantile', quantile, epoch)

        result = {'quantile': quantile, 'scope': scope}
        if wass_pos is not None:
            result['wasserstein_pos'] = wass_pos
        if wass_neg is not None:
            result['wasserstein_neg'] = wass_neg
        if wass_diff is not None:
            result['wasserstein_diff'] = wass_diff
        return result

    def _factset_edge_scores(self, epoch=None, max_factset_edges=None):
        """Score the FactSet edges, cached per epoch so both scopes share one inference pass."""
        cacheable = epoch is not None
        if cacheable and self._factset_cache_epoch == epoch \
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
            batch_size = 256
            with torch.no_grad():
                for i in tqdm(range(0, len(valid_pairs), batch_size),
                              desc="Factset: scoring edges", leave=False):
                    batch_pairs = torch.tensor(valid_pairs[i:i+batch_size], dtype=torch.long,
                                               device=self.device)
                    batch_times = torch.tensor(valid_years[i:i+batch_size], dtype=torch.float,
                                               device=self.device)
                    preds = self._predict(batch_pairs, batch_times)
                    if len(preds) > 0:
                        factset_scores.extend(preds.cpu().numpy().flatten().tolist())

        if cacheable:
            self._factset_cache_epoch = epoch
            self._factset_cache = (max_factset_edges, factset_scores)
        return factset_scores

    def _split_scores(self, loader, scope):
        """All / positive / negative scores of one split.

        Reuses the cached evaluate() pass when available, so the FactSet statistics describe
        exactly the same sample set as the reported metrics instead of a second inference pass.
        """
        cached = self._eval_scores.get(scope)
        if cached is not None:
            preds, labels = cached
            preds_np = preds.detach().cpu().numpy().flatten()
            labels_np = labels.detach().cpu().numpy().flatten()
            return (preds_np.tolist(),
                    preds_np[labels_np == 1].tolist(),
                    preds_np[labels_np == 0].tolist())

        scores_all, pos_scores, neg_scores = [], [], []
        with torch.no_grad():
            for link_indices, current_times, labels in tqdm(
                    loader, desc=f"Factset: scoring {scope} set", leave=False):
                link_indices = link_indices.to(self.device)
                current_times = current_times.to(self.device)
                predictions = self._predict(link_indices, current_times)
                if len(predictions) > 0:
                    preds_np = predictions.cpu().numpy().flatten()
                    labels_np = labels.cpu().numpy().flatten()
                    scores_all.extend(preds_np.tolist())
                    pos_scores.extend(preds_np[labels_np == 1].tolist())
                    neg_scores.extend(preds_np[labels_np == 0].tolist())
        return scores_all, pos_scores, neg_scores

    @staticmethod
    def _fmt_factset_report(result, split_label):
        """Console lines for one split's FactSet statistics."""
        if result is None:
            return []
        lines = [f"Factset quantile ({split_label}): {result['quantile']:.4f} (higher is better)"]
        if 'wasserstein_pos' in result:
            lines.append(f"Wasserstein ({split_label}, Factset vs positives): "
                         f"{result['wasserstein_pos']:.6f} (smaller is more similar)")
        if 'wasserstein_neg' in result:
            lines.append(f"Wasserstein ({split_label}, Factset vs negatives): "
                         f"{result['wasserstein_neg']:.6f} (smaller is more similar)")
        if 'wasserstein_diff' in result:
            lines.append(f"Wasserstein Diff ({split_label}, neg-pos): "
                         f"{result['wasserstein_diff']:.6f} (larger is better)")
        return lines

    @staticmethod
    def _factset_ckpt_fields(factset_result):
        """Checkpoint fields for the selection-split and test-split FactSet statistics."""
        fields = {}
        if factset_result is None:
            return fields
        fields['factset_scope'] = factset_result.get('scope', 'test')
        for key in ('quantile', 'wasserstein_pos', 'wasserstein_neg', 'wasserstein_diff'):
            if key in factset_result:
                fields['factset_' + key] = factset_result[key]
        test_result = factset_result.get('test')
        if test_result is not None:
            for key in ('quantile', 'wasserstein_pos', 'wasserstein_neg', 'wasserstein_diff'):
                if key in test_result:
                    fields['factset_' + key + '_test'] = test_result[key]
        return fields

    def train_epoch(self, epoch, save_path):
        self.model.train()
        total_bce_loss = 0
        total_margin_loss = 0
        all_predictions = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training')

        for batch_idx, (link_indices, current_times, labels) in enumerate(pbar):
            link_indices = link_indices.to(self.device)
            current_times = current_times.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            predictions = self._predict(link_indices, current_times)

            if len(predictions) <= 1:
                continue

            bce_loss = self.bce_criterion(predictions, labels)

            # Margin Ranking Loss (pairwise ranking constraint)
            margin_loss = torch.tensor(0.0, device=self.device)
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_scores = predictions[pos_mask]
                neg_scores = predictions[neg_mask]
                # Compare each positive against all negatives, forcing positive score > negative score
                pos_expanded = pos_scores.repeat_interleave(len(neg_scores))
                neg_expanded = neg_scores.repeat(len(pos_scores))
                target = torch.ones_like(pos_expanded)  # expect pos > neg
                margin_loss = self.margin_loss(pos_expanded, neg_expanded, target)

            loss = bce_loss + self.margin_lambda * margin_loss
            loss.backward()
            self.optimizer.step()

            total_bce_loss += bce_loss.item()
            total_margin_loss += margin_loss.item()
            all_predictions.append(predictions.detach())
            all_labels.append(labels.detach())

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0 and len(all_predictions) >= 10:
                batch_predictions = torch.cat(all_predictions[-10:])
                batch_labels = torch.cat(all_labels[-10:])
                precision, recall, f1, auc = self.compute_metrics(batch_predictions, batch_labels)

                pbar.set_postfix({
                    'BCE': f'{bce_loss.item():.4f}',
                    'Margin': f'{margin_loss.item():.4f}',
                    'F1': f'{f1:.4f}',
                })

                if self.use_tensorboard:
                    step = self.global_train_step
                    self.writer.add_scalar('Train/Batch_BCE', bce_loss.item(), step)
                    self.writer.add_scalar('Train/Batch_Margin', margin_loss.item(), step)
                    self.writer.add_scalar('Train/Batch_F1', f1, step)
                    self.writer.add_scalar('Train/Batch_AUC', auc, step)
                    self.global_train_step += 1

        if len(all_predictions) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        avg_bce = total_bce_loss / len(self.train_loader)
        avg_margin = total_margin_loss / len(self.train_loader)
        precision, recall, f1, auc = self.compute_metrics(all_predictions, all_labels)

        if self.use_tensorboard:
            self.writer.add_scalar('Train/Epoch_BCE', avg_bce, epoch)
            self.writer.add_scalar('Train/Epoch_Margin', avg_margin, epoch)
            self.writer.add_scalar('Train/Epoch_F1', f1, epoch)
            self.writer.add_scalar('Train/Epoch_AUC', auc, epoch)

        return avg_bce, avg_margin, precision, recall, f1, auc

    def evaluate(self, data_loader, epoch, mode='val'):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f'Epoch {epoch+1} {mode.capitalize()} Evaluation')

            for batch_idx, (link_indices, current_times, labels) in enumerate(pbar):
                link_indices = link_indices.to(self.device)
                current_times = current_times.to(self.device)
                labels = labels.to(self.device).float()

                predictions = self._predict(link_indices, current_times)
                if len(predictions) <= 1:
                    continue

                loss = self.bce_criterion(predictions, labels)
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_labels.append(labels)

                if (batch_idx + 1) % 10 == 0 and len(all_predictions) >= 10:
                    batch_predictions = torch.cat(all_predictions[-10:])
                    batch_labels = torch.cat(all_labels[-10:])
                    precision, recall, f1, auc = self.compute_metrics(batch_predictions, batch_labels)

                    if self.use_tensorboard:
                        step_attr = f'global_{mode}_step'
                        global_step = getattr(self, step_attr)
                        self.writer.add_scalar(f'{mode.capitalize()}/Batch_Loss', loss.item(), global_step)
                        self.writer.add_scalar(f'{mode.capitalize()}/Batch_F1', f1, global_step)
                        self.writer.add_scalar(f'{mode.capitalize()}/Batch_AUC', auc, global_step)
                        setattr(self, step_attr, global_step + 1)

        if len(all_predictions) == 0:
            self._eval_scores.pop(mode, None)
            return 0.0, 0.0, 0.0, 0.0, 0.0

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        avg_loss = total_loss / len(data_loader)
        precision, recall, f1, auc = self.compute_metrics(all_predictions, all_labels)

        # Cache this pass so the FactSet statistics of the same split reuse it instead of running
        # a second inference pass (which would also give slightly different scores under
        # non-deterministic CUDA kernels).
        self._eval_scores[mode] = (all_predictions, all_labels)

        if self.use_tensorboard:
            self.writer.add_scalar(f'{mode.capitalize()}/Epoch_Loss', avg_loss, epoch)
            self.writer.add_scalar(f'{mode.capitalize()}/Epoch_F1', f1, epoch)
            self.writer.add_scalar(f'{mode.capitalize()}/Epoch_AUC', auc, epoch)

        return avg_loss, precision, recall, f1, auc

    def save_test_predictions_npy(self, npy_path, checkpoint_path=None):
        """
        Run inference on the test set and save prediction scores as .npy (dict:
        'test_predictions'/'neg_predictions', each a list of positive/negative sample scores),
        for ROC/AUC curve plotting.
        """
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
                predictions = self._predict(link_indices, current_times)
                if len(predictions) > 0:
                    all_predictions.append(predictions.cpu().numpy().flatten())
                    all_labels.append(labels.cpu().numpy().flatten())

        if len(all_predictions) == 0:
            print("[SavePred] WARNING: no valid test-set predictions, skipping save")
            return

        all_preds = np.concatenate(all_predictions)
        all_labs = np.concatenate(all_labels)

        # Separate positive and negative sample scores
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

    def train(self, num_epochs, save_path='best_dynamic_graph_model.pth', patience=10,
              max_factset_edges=None):
        best_f1 = 0
        best_epoch = 0
        best_auc = 0.0
        patience_counter = 0

        print("Starting dynamic graph imputation model training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset) if self.val_loader else 0}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        factset_display = len(self.factset_edges)
        if max_factset_edges and factset_display > max_factset_edges:
            factset_display = max_factset_edges
        print(f"Factset edges: {len(self.factset_edges)} (actually used: {factset_display})")
        print(f"Margin Lambda: {self.margin_lambda}")

        try:
            for epoch in range(num_epochs):
                epoch_start_time = time.time()

                train_bce, train_margin, train_prec, train_rec, train_f1, train_auc = \
                    self.train_epoch(epoch, save_path=save_path)

                val_loss = val_prec = val_rec = val_f1 = val_auc = 0.0
                if self.val_loader:
                    val_loss, val_prec, val_rec, val_f1, val_auc = self.evaluate(self.val_loader, epoch, 'val')

                test_loss, test_prec, test_rec, test_f1, test_auc = self.evaluate(self.test_loader, epoch, 'test')

                # FactSet statistics are computed on the split that drives model selection, plus a
                # test-split copy for reporting.
                factset_result = self.compute_factset_quantile(
                    epoch=epoch, max_factset_edges=max_factset_edges)

                epoch_duration = time.time() - epoch_start_time
                self.epoch_durations.append(epoch_duration)
                if self.use_tensorboard:
                    self.writer.add_scalar('Time/Epoch_Duration', epoch_duration, epoch)

                print(f"\nEpoch {epoch+1} summary ({epoch_duration:.1f}s):")
                print(f"Train - BCE: {train_bce:.4f}, Margin: {train_margin:.4f}, "
                      f"Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, "
                      f"F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
                if self.val_loader:
                    print(f"Val   - Loss: {val_loss:.4f}, Precision: {val_prec:.4f}, "
                          f"Recall: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
                print(f"Test  - Loss: {test_loss:.4f}, Precision: {test_prec:.4f}, "
                      f"Recall: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

                if factset_result is not None:
                    for line in self._fmt_factset_report(factset_result,
                                                         self.selection_split.capitalize()):
                        print(line)
                    for line in self._fmt_factset_report(factset_result.get('test'), 'Test'):
                        print(line)

                # Model selection: the validation split when one is configured, the test split
                # otherwise. The composite score is
                #   0.5 * FactSet quantile + 0.5 * AUC(selection split).
                select_auc = val_auc if self.val_loader is not None else test_auc
                select_f1 = val_f1 if self.val_loader is not None else test_f1
                current_score = select_f1
                if factset_result is not None:
                    factset_q = factset_result['quantile']
                    current_score = 0.5 * factset_q + 0.5 * select_auc

                test_factset = factset_result.get('test') if factset_result is not None else None
                self.history.append({
                    'epoch': epoch,
                    'selection_split': self.selection_split,
                    'train_bce': train_bce,
                    'train_margin': train_margin,
                    'train_f1': train_f1,
                    'train_auc': train_auc,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'val_auc': val_auc,
                    'test_loss': test_loss,
                    'test_f1': test_f1,
                    'test_auc': test_auc,
                    'factset_quantile': factset_result['quantile'] if factset_result else None,
                    'wasserstein_diff': factset_result.get('wasserstein_diff')
                    if factset_result else None,
                    'factset_quantile_test': test_factset.get('quantile') if test_factset else None,
                    'wasserstein_diff_test': test_factset.get('wasserstein_diff')
                    if test_factset else None,
                    'epoch_seconds': epoch_duration,
                    'score': current_score,
                })

                if current_score > best_f1:
                    best_f1 = current_score
                    best_epoch = epoch
                    patience_counter = 0
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'selection_split': self.selection_split,
                        'train_bce': train_bce,
                        'train_margin': train_margin,
                        'epoch_seconds': epoch_duration,
                        'val_loss': val_loss,
                        'val_f1': val_f1,
                        'val_auc': val_auc,
                        'test_loss': test_loss,
                        'test_f1': test_f1,
                        'test_auc': test_auc,
                    }
                    save_dict.update(self._factset_ckpt_fields(factset_result))
                    torch.save(save_dict, save_path)
                    print(f"Saved best model (epoch {epoch+1}), composite score: {current_score:.4f} "
                          f"({self.selection_split} split)")
                    best_npy_path = os.path.join(os.path.dirname(save_path), 'model_predictions_best.npy')
                    self.save_test_predictions_npy(best_npy_path, checkpoint_path=None)

                # Additionally track the epoch with the highest AUC of the selection split
                if select_auc > best_auc:
                    best_auc = select_auc
                    auc_save_path = os.path.join(os.path.dirname(save_path), 'best_auc_model.pth')
                    auc_save_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'selection_split': self.selection_split,
                        'train_bce': train_bce,
                        'train_margin': train_margin,
                        'epoch_seconds': epoch_duration,
                        'val_loss': val_loss,
                        'val_f1': val_f1,
                        'val_auc': val_auc,
                        'test_loss': test_loss,
                        'test_f1': test_f1,
                        'test_auc': test_auc,
                    }
                    auc_save_dict.update(self._factset_ckpt_fields(factset_result))
                    torch.save(auc_save_dict, auc_save_path)
                    print(f"Saved best-AUC model (epoch {epoch+1}), "
                          f"{self.selection_split} AUC: {select_auc:.4f}")
                    best_auc_npy_path = os.path.join(os.path.dirname(save_path), 'model_predictions_best_auc.npy')
                    self.save_test_predictions_npy(best_auc_npy_path, checkpoint_path=None)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered! Stopping training at epoch {epoch+1}")
                        break

        finally:
            if self.use_tensorboard:
                self.writer.close()

        print(f"\nTraining complete! Best epoch: {best_epoch+1}, best score: {best_f1:.4f} "
              f"({self.selection_split} split)")
        if self.epoch_durations:
            print(f"Average epoch time: {float(np.mean(self.epoch_durations)):.1f}s "
                  f"over {len(self.epoch_durations)} epochs")
        return best_epoch, best_f1
