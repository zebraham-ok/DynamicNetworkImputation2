"""
General-purpose trainer: supports BCE + Margin Ranking Loss (pairwise ranking loss),
Factset quantile monitoring, and AUC.
Usage:
    trainer = DynamicGraphTrainer(
        model=model, train_loader=train_loader, test_loader=test_loader,
        factset_edges=factset_edges,  # optional: [(src, tgt, year), ...] for the factset quantile metric
        margin_lambda=0.1,            # Margin Ranking Loss weight
    )
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class DynamicGraphTrainer:
    def __init__(self, model, train_loader, test_loader, val_loader=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 log_dir=None, use_tensorboard=True,
                 margin_lambda=0.1, margin=1.0, factset_edges=None,
                 node_mapping=None, reverse_node_mapping=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

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

    def compute_factset_quantile(self, epoch=None, max_factset_edges=None):
        """
        Compute the mean quantile of factset edges within the test-set scores, and the
        Wasserstein-1 distance between the factset score distribution and the test-set
        positive/negative examples.
        Higher quantile / smaller Wasserstein distance -> the model better captures
        true supply relationships.
        Returns dict {'quantile': float, 'wasserstein_pos': float, 'wasserstein_neg': float} or None.
        """
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
        batch_size = 256
        with torch.no_grad():
            for i in tqdm(range(0, len(valid_pairs), batch_size), desc="Factset: scoring edges", leave=False):
                batch_pairs = torch.tensor(valid_pairs[i:i+batch_size], dtype=torch.long, device=self.device)
                batch_times = torch.tensor(valid_years[i:i+batch_size], dtype=torch.float, device=self.device)
                preds = self.model(batch_pairs, batch_times)
                if len(preds) > 0:
                    factset_scores.extend(preds.cpu().numpy().flatten().tolist())

        if not factset_scores:
            return None

        # Scores of all test-set edges (separating positive/negative examples)
        test_scores = []
        test_pos_scores = []
        test_neg_scores = []
        with torch.no_grad():
            for link_indices, current_times, labels in tqdm(self.test_loader, desc="Factset: scoring test set", leave=False):
                link_indices = link_indices.to(self.device)
                current_times = current_times.to(self.device)
                predictions = self.model(link_indices, current_times)
                if len(predictions) > 0:
                    preds_np = predictions.cpu().numpy().flatten()
                    test_scores.extend(preds_np.tolist())
                    labels_np = labels.cpu().numpy().flatten()
                    test_pos_scores.extend(preds_np[labels_np == 1].tolist())
                    test_neg_scores.extend(preds_np[labels_np == 0].tolist())

        if not test_scores:
            return None

        if self.use_tensorboard and epoch is not None:
            factset_arr = np.array(factset_scores)
            test_arr = np.array(test_scores)
            
            self.writer.add_histogram('Score_Distribution/TestSet_All', test_arr, epoch)
            self.writer.add_histogram('Score_Distribution/Factset', factset_arr, epoch)
            
            if test_pos_scores:
                self.writer.add_histogram('Score_Distribution/TestSet_Positive', np.array(test_pos_scores), epoch)
            if test_neg_scores:
                self.writer.add_histogram('Score_Distribution/TestSet_Negative', np.array(test_neg_scores), epoch)

        # Wasserstein-1 distance: between the factset score distribution and the test-set positive/negative distributions
        factset_arr = np.array(factset_scores)
        wass_pos = None
        wass_neg = None
        if test_pos_scores:
            wass_pos = wasserstein_distance(factset_arr, np.array(test_pos_scores))
        if test_neg_scores:
            wass_neg = wasserstein_distance(factset_arr, np.array(test_neg_scores))

        if self.use_tensorboard and epoch is not None:
            if wass_pos is not None:
                self.writer.add_scalar('Monitor/Wasserstein_Factset_vs_Pos', wass_pos, epoch)
            if wass_neg is not None:
                self.writer.add_scalar('Monitor/Wasserstein_Factset_vs_Neg', wass_neg, epoch)

        # Wasserstein difference (negative distance - positive distance, larger is better)
        wass_diff = None
        if wass_pos is not None and wass_neg is not None:
            wass_diff = wass_neg - wass_pos
            if self.use_tensorboard and epoch is not None:
                self.writer.add_scalar('Monitor/Wasserstein_Diff', wass_diff, epoch)

        # Compute quantiles
        test_scores_sorted = np.sort(test_scores)
        percentiles = [np.searchsorted(test_scores_sorted, s) / len(test_scores_sorted)
                       for s in factset_scores]
        quantile = float(np.mean(percentiles))

        # Return a dict so callers can obtain all metrics
        result = {'quantile': quantile}
        if wass_pos is not None:
            result['wasserstein_pos'] = wass_pos
        if wass_neg is not None:
            result['wasserstein_neg'] = wass_neg
        if wass_diff is not None:
            result['wasserstein_diff'] = wass_diff
        return result

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
            predictions = self.model(link_indices, current_times)

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

                predictions = self.model(link_indices, current_times)
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
            return 0.0, 0.0, 0.0, 0.0, 0.0

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        avg_loss = total_loss / len(data_loader)
        precision, recall, f1, auc = self.compute_metrics(all_predictions, all_labels)

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
                predictions = self.model(link_indices, current_times)
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

                factset_result = self.compute_factset_quantile(epoch=epoch, max_factset_edges=max_factset_edges)

                epoch_duration = time.time() - epoch_start_time
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
                    factset_q = factset_result['quantile']
                    print(f"Factset quantile: {factset_q:.4f} (higher is better)")
                    if self.use_tensorboard:
                        self.writer.add_scalar('Monitor/Factset_Quantile', factset_q, epoch)
                    if 'wasserstein_pos' in factset_result:
                        w_pos = factset_result['wasserstein_pos']
                        print(f"Wasserstein (Factset vs Test positives): {w_pos:.6f} (smaller is more similar)")
                    if 'wasserstein_neg' in factset_result:
                        w_neg = factset_result['wasserstein_neg']
                        print(f"Wasserstein (Factset vs Test negatives): {w_neg:.6f} (smaller is more similar)")
                    if 'wasserstein_diff' in factset_result:
                        w_diff = factset_result['wasserstein_diff']
                        print(f"Wasserstein Diff (neg-pos): {w_diff:.6f} (larger is better)")

                # Select best model: with factset use 0.5*factset quantile + 0.5*AUC, else val_f1 (or test_f1)
                current_score = test_f1 if self.val_loader is None else val_f1
                if factset_result is not None:
                    factset_q = factset_result['quantile']
                    current_score = 0.5 * factset_q + 0.5 * test_auc

                if current_score > best_f1:
                    best_f1 = current_score
                    best_epoch = epoch
                    patience_counter = 0
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_bce': train_bce,
                        'test_f1': test_f1,
                        'test_auc': test_auc,
                    }
                    if factset_result is not None:
                        save_dict['factset_quantile'] = factset_result['quantile']
                        if 'wasserstein_pos' in factset_result:
                            save_dict['wasserstein_pos'] = factset_result['wasserstein_pos']
                        if 'wasserstein_neg' in factset_result:
                            save_dict['wasserstein_neg'] = factset_result['wasserstein_neg']
                        if 'wasserstein_diff' in factset_result:
                            save_dict['wasserstein_diff'] = factset_result['wasserstein_diff']
                    torch.save(save_dict, save_path)
                    print(f"Saved best model (epoch {epoch+1}), composite score: {current_score:.4f}")
                    best_npy_path = os.path.join(os.path.dirname(save_path), 'model_predictions_best.npy')
                    self.save_test_predictions_npy(best_npy_path, checkpoint_path=None)

                # Additionally track the epoch with the highest AUC over the whole training run
                if test_auc > best_auc:
                    best_auc = test_auc
                    auc_save_path = os.path.join(os.path.dirname(save_path), 'best_auc_model.pth')
                    auc_save_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_bce': train_bce,
                        'test_f1': test_f1,
                        'test_auc': test_auc,
                    }
                    if factset_result is not None:
                        auc_save_dict['factset_quantile'] = factset_result['quantile']
                        if 'wasserstein_pos' in factset_result:
                            auc_save_dict['wasserstein_pos'] = factset_result['wasserstein_pos']
                        if 'wasserstein_neg' in factset_result:
                            auc_save_dict['wasserstein_neg'] = factset_result['wasserstein_neg']
                        if 'wasserstein_diff' in factset_result:
                            auc_save_dict['wasserstein_diff'] = factset_result['wasserstein_diff']
                    torch.save(auc_save_dict, auc_save_path)
                    print(f"Saved best-AUC model (epoch {epoch+1}), AUC: {test_auc:.4f}")
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

        print(f"\nTraining complete! Best epoch: {best_epoch+1}, best score: {best_f1:.4f}")
        return best_epoch, best_f1
