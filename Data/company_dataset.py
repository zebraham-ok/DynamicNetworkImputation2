"""
Dataset module (sole entry point of the Data layer): three-way split + dynamic negative sampling with blacklist

Split protocol:
    - Held-out test set (Test): 10% positives + equal fixed negatives (permanently frozen)
    - Fixed validation set (Val): 10% positives + equal fixed negatives (permanently frozen)
    - Train pool: remaining 80% positives (negatives generated dynamically each iteration)

Public interface:
    - create_bootstrap_datasets()  three-way split (full / test / val / train_pool)
    - create_dataloaders()         standard training 5-tuple (static, train, val, test, full)
    - BootstrapIterationDataset    training dataset for a single bootstrap iteration
    - load_pretrained_backbone() / reinit_trainable_parts()  freeze / reset backbone

Low-level primitives (CompanySupplyDataset, build_static_graph) are defined in `Data.graph_dataset`
and re-exported here; config's `dataset.module` is uniformly set to `Data.company_dataset`.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.data import Data

from Data.graph_dataset import CompanySupplyDataset, build_static_graph


def create_bootstrap_datasets(
    negative_ratio: int = 1,
    embedding_name: str = "embedding",
    test_ratio: float = 0.10,
    val_ratio: float = 0.10,
    random_state: int = 42,
    min_degree: int = 2,
    source_filter: str = 'semi',
    other_possible_fill: float = 0.0,
    filter_factset_neg: bool = False,
    intra_industry_neg: bool = True,
    toy_mode: bool = False,
):
    """Create the three-way bootstrap data split (full / test / val / train_pool)."""
    full_dataset = CompanySupplyDataset(
        negative_ratio=0,
        embedding_name=embedding_name,
        neg_dir=None,
        toy_mode=toy_mode,
        min_degree=min_degree,
        source_filter=source_filter,
        other_possible_fill=other_possible_fill,
        filter_factset_neg=filter_factset_neg,
        intra_industry_neg=intra_industry_neg,
        use_pred_neg=False,
    )

    # Stratified split by year (dedup first, to avoid the same triple landing in different sets and leaking)
    all_pos_raw = full_dataset.original_positive_samples
    n_raw = len(all_pos_raw)
    all_pos = list(set(all_pos_raw))
    n_dup = n_raw - len(all_pos)
    if n_dup > 0:
        print(f"\n[Bootstrap Data] Detected {n_dup} duplicate triples, deduplicated")
    all_pos.sort(key=lambda s: (s[2], s[0], s[1]))  # sort by (year, src, tgt) for determinism
    years = [s[2] for s in all_pos]
    print(f"[Bootstrap Data] Total positives: {len(all_pos)}" +
          (f" (raw {n_raw}, dedup {n_dup})" if n_dup > 0 else ""))

    # First carve out test (10%)
    remaining_pos, test_pos = train_test_split(
        all_pos, test_size=test_ratio,
        random_state=random_state, stratify=years
    )
    remaining_years = [s[2] for s in remaining_pos]

    val_size = val_ratio / (1 - test_ratio)
    train_pool_pos, val_pos = train_test_split(
        remaining_pos, test_size=val_size,
        random_state=random_state, stratify=remaining_years
    )

    print(f"[Bootstrap Data] Test:  {len(test_pos)} ({len(test_pos)/len(all_pos)*100:.1f}%)")
    print(f"[Bootstrap Data] Val:   {len(val_pos)} ({len(val_pos)/len(all_pos)*100:.1f}%)")
    print(f"[Bootstrap Data] Train Pool: {len(train_pool_pos)} ({len(train_pool_pos)/len(all_pos)*100:.1f}%)")

    # Generate fixed negatives for test/val and build the blacklist
    rng = np.random.RandomState(random_state)
    neg_pool = _build_negative_pool(full_dataset, all_pos, rng)

    test_neg = _sample_fixed_negatives(
        full_dataset, test_pos, len(test_pos), neg_pool, rng
    )
    val_neg = _sample_fixed_negatives(
        full_dataset, val_pos, len(val_pos), neg_pool, rng
    )

    # Blacklist: all positives and negatives in test/val are barred from training
    blacklist = set()
    for s in test_pos + val_pos:
        blacklist.add(s)
    for s in test_neg + val_neg:
        blacklist.add(s)

    print(f"[Bootstrap Data] Blacklist size: {len(blacklist)} "
          f"(test pos {len(test_pos)}+test neg {len(test_neg)}+val pos {len(val_pos)}+val neg {len(val_neg)})")

    # Validate year range: all positives and available_years must fall within 2013-2025
    _YEAR_MIN, _YEAR_MAX = 2013, 2025
    for label, samples in [("test_pos", test_pos), ("val_pos", val_pos), ("train_pool", train_pool_pos)]:
        out_of_range = [(s, t, y) for s, t, y in samples if y < _YEAR_MIN or y > _YEAR_MAX]
        if out_of_range:
            raise ValueError(
                f"[Bootstrap Data] {label} contains {len(out_of_range)} samples with year outside {_YEAR_MIN}-{_YEAR_MAX}!"
                f" Examples: {out_of_range[:3]}"
            )
    out_of_range_years = [y for y in full_dataset.available_years if y < _YEAR_MIN or y > _YEAR_MAX]
    if out_of_range_years:
        raise ValueError(
            f"[Bootstrap Data] available_years contains years outside {_YEAR_MIN}-{_YEAR_MAX}: {out_of_range_years}"
        )
    print(f"[Bootstrap Data] Year range validation passed: all data within {_YEAR_MIN}-{_YEAR_MAX} (available_years={full_dataset.available_years})")

    # Create test/val datasets (fixed negatives)
    test_dataset = CompanySupplyDataset(
        negative_ratio=negative_ratio,
        embedding_name=embedding_name,
        neg_dir=None,
        toy_mode=toy_mode,
        positive_samples=list(test_pos),
        predifined_neg_data=test_neg,
        min_degree=min_degree,
        source_filter=source_filter,
        other_possible_fill=other_possible_fill,
        filter_factset_neg=filter_factset_neg,
        intra_industry_neg=False,
        use_pred_neg=True,
    )
    _share_metadata(test_dataset, full_dataset)

    val_dataset = CompanySupplyDataset(
        negative_ratio=negative_ratio,
        embedding_name=embedding_name,
        neg_dir=None,
        toy_mode=toy_mode,
        positive_samples=list(val_pos),
        predifined_neg_data=val_neg,
        min_degree=min_degree,
        source_filter=source_filter,
        other_possible_fill=other_possible_fill,
        filter_factset_neg=filter_factset_neg,
        intra_industry_neg=False,
        use_pred_neg=True,
    )
    _share_metadata(val_dataset, full_dataset)

    # train_pool_dataset holds metadata only, not used for training
    train_pool_dataset = CompanySupplyDataset(
        negative_ratio=0,
        embedding_name=embedding_name,
        neg_dir=None,
        toy_mode=toy_mode,
        positive_samples=list(train_pool_pos),
        predifined_neg_data=[],
        min_degree=min_degree,
        source_filter=source_filter,
        other_possible_fill=other_possible_fill,
        filter_factset_neg=filter_factset_neg,
        intra_industry_neg=False,
        use_pred_neg=False,
    )
    _share_metadata(train_pool_dataset, full_dataset)

    return {
        'full_dataset': full_dataset,
        'test_dataset': test_dataset,
        'val_dataset': val_dataset,
        'train_pool': list(train_pool_pos),
        'train_pool_dataset': train_pool_dataset,
        'blacklist': blacklist,
        'company_ids': full_dataset.company_ids,
        'company2industry': full_dataset.company2industry,
        'industry2company': full_dataset.industry2company,
        'node_mapping': full_dataset.node_mapping,
        'reverse_node_mapping': full_dataset.reverse_node_mapping,
        'available_years': full_dataset.available_years,
    }


def _share_metadata(dataset, source):
    """Share metadata"""
    dataset.node_mapping = source.node_mapping
    dataset.reverse_node_mapping = source.reverse_node_mapping
    dataset.factset_edges = source.factset_edges


# Study-window year range (global constants)
_STUDY_YEAR_MIN = 2013
_STUDY_YEAR_MAX = 2025


def _clamp_year(year: int, available_years: list) -> int:
    """Clamp a year to the study window; if out of range, pick one at random from available_years"""
    if _STUDY_YEAR_MIN <= year <= _STUDY_YEAR_MAX:
        return year
    candidates = [y for y in available_years if _STUDY_YEAR_MIN <= y <= _STUDY_YEAR_MAX]
    if candidates:
        return int(np.random.choice(candidates))
    return int(np.random.randint(_STUDY_YEAR_MIN, _STUDY_YEAR_MAX + 1))


def _build_negative_pool(full_dataset, all_pos, rng):
    """Build the global negative candidate pool (to avoid regenerating identical negatives)
    
    All negative-sample years are restricted to the {_STUDY_YEAR_MIN}-{_STUDY_YEAR_MAX} study window.
    """
    known_edges = set(all_pos)
    if full_dataset.factset_edges:
        known_edges.update(full_dataset.factset_edges)

    company_ids = full_dataset.company_ids
    available_years = [y for y in full_dataset.available_years
                       if _STUDY_YEAR_MIN <= y <= _STUDY_YEAR_MAX]
    # Pre-generate a negative pool 10x the size
    pool_size = len(all_pos) * 10
    pool = set()
    max_attempts = pool_size * 10
    attempts = 0

    while len(pool) < pool_size and attempts < max_attempts:
        src = company_ids[rng.randint(0, len(company_ids))]
        tgt = company_ids[rng.randint(0, len(company_ids))]
        year = available_years[rng.randint(0, len(available_years))]
        triple = (src, tgt, year)
        if triple not in known_edges and triple not in pool:
            pool.add(triple)
        attempts += 1

    return list(pool)


def _sample_fixed_negatives(full_dataset, pos_samples, neg_count, neg_pool, rng):
    """Sample a fixed number of negatives from the pool (years restricted to {_STUDY_YEAR_MIN}-{_STUDY_YEAR_MAX})"""
    known_edges = set(pos_samples)
    if full_dataset.factset_edges:
        known_edges.update(full_dataset.factset_edges)

    available = [n for n in neg_pool if n not in known_edges]
    if len(available) >= neg_count:
        idx = rng.choice(len(available), neg_count, replace=False)
        return [available[i] for i in idx]
    else:
        result = list(available)
        company_ids = full_dataset.company_ids
        available_years = [y for y in full_dataset.available_years
                           if _STUDY_YEAR_MIN <= y <= _STUDY_YEAR_MAX]
        while len(result) < neg_count:
            src = company_ids[rng.randint(0, len(company_ids))]
            tgt = company_ids[rng.randint(0, len(company_ids))]
            year = available_years[rng.randint(0, len(available_years))]
            triple = (src, tgt, year)
            if triple not in known_edges and triple not in result:
                result.append(triple)
        return result


class BootstrapIterationDataset(Dataset):
    """Training dataset for a single bootstrap iteration: sample positives from train_pool and dynamically generate an equal number of negatives (avoiding the blacklist)."""

    def __init__(
        self,
        train_pool: List[Tuple],
        blacklist: Set[Tuple],
        meta_dataset: CompanySupplyDataset,
        negative_ratio: int = 1,
        seed: int = 0,
        intra_industry_neg: bool = True,
        bootstrap: bool = True,
    ):
        self.meta = meta_dataset
        self.negative_ratio = negative_ratio
        self.intra_industry_neg = intra_industry_neg

        rng = np.random.RandomState(seed)

        if bootstrap:
            n = len(train_pool)
            indices = rng.randint(0, n, size=n)
            self.bootstrapped_pos = [train_pool[i] for i in indices]
        else:
            self.bootstrapped_pos = list(train_pool)

        all_known = blacklist | set(self.bootstrapped_pos)
        if hasattr(meta_dataset, 'factset_edges') and meta_dataset.factset_edges:
            all_known.update(meta_dataset.factset_edges)

        self.dynamic_negatives = _generate_bootstrap_negatives(
            meta_dataset, self.bootstrapped_pos, all_known, rng, intra_industry_neg
        )

        self.positive_samples = list(self.bootstrapped_pos)
        self.total_pos = len(self.positive_samples)
        self.total_neg = len(self.dynamic_negatives)

        self._known_edges_cache = all_known

        print(f"[Bootstrap Iter seed={seed}] positives: {self.total_pos} (unique: {len(set(self.bootstrapped_pos))}), negatives: {self.total_neg}")

    def __len__(self):
        return self.total_pos + self.total_neg

    def __getitem__(self, idx):
        node_mapping = self.meta.node_mapping

        if idx < self.total_pos:
            source_id, target_id, year = self.positive_samples[idx]
            label = 1.0
        else:
            neg_idx = idx - self.total_pos
            source_id, target_id, year = self.dynamic_negatives[neg_idx]
            label = 0.0

        source_idx = node_mapping[source_id]
        target_idx = node_mapping[target_id]

        link_indices = torch.tensor([source_idx, target_idx], dtype=torch.long)
        current_time = torch.tensor(year, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return link_indices, current_time, label


def _generate_bootstrap_negatives(
    meta_dataset, pos_samples, blacklist, rng, intra_industry_neg
):
    """Generate dynamic negatives for a bootstrap iteration"""
    company_ids = meta_dataset.company_ids
    available_years = meta_dataset.available_years
    neg_count = len(pos_samples)

    if intra_industry_neg and hasattr(meta_dataset, 'company2industry') and meta_dataset.company2industry:
        company2industry = meta_dataset.company2industry
        industry2company = meta_dataset.industry2company
        return _generate_intra_industry_negatives(
            pos_samples, neg_count, company_ids, available_years,
            company2industry, industry2company, blacklist, rng
        )
    else:
        return _generate_random_negatives(
            pos_samples, neg_count, company_ids, available_years, blacklist, rng
        )


def _generate_random_negatives(pos_samples, neg_count, company_ids, available_years, blacklist, rng):
    """Purely random negative sampling (years restricted to {_STUDY_YEAR_MIN}-{_STUDY_YEAR_MAX})"""
    # Ensure years are within the study window
    available_years = [y for y in available_years
                       if _STUDY_YEAR_MIN <= y <= _STUDY_YEAR_MAX]
    negatives = []
    n_companies = len(company_ids)
    n_years = len(available_years)
    max_attempts = neg_count * 20
    attempts = 0

    while len(negatives) < neg_count and attempts < max_attempts:
        src = company_ids[rng.randint(0, n_companies)]
        tgt = company_ids[rng.randint(0, n_companies)]
        year = available_years[rng.randint(0, n_years)]
        triple = (src, tgt, year)
        if triple not in blacklist and triple not in negatives:
            negatives.append(triple)
        attempts += 1

    return negatives


def _generate_intra_industry_negatives(
    pos_samples, neg_count, company_ids, available_years,
    company2industry, industry2company, blacklist, rng
):
    """Intra-industry negative sampling (years restricted to {_STUDY_YEAR_MIN}-{_STUDY_YEAR_MAX})"""
    # Ensure years are within the study window
    available_years = [y for y in available_years
                       if _STUDY_YEAR_MIN <= y <= _STUDY_YEAR_MAX]
    negatives = []
    n_years = len(available_years)
    n_companies = len(company_ids)
    max_attempts = neg_count * 20
    attempts = 0

    # Pre-build an industry2company list version to speed up random access
    ind2comp_lists = {k: list(v) for k, v in industry2company.items()}

    while len(negatives) < neg_count and attempts < max_attempts:
        # Pick a random positive as the "seed"
        seed_idx = rng.randint(0, len(pos_samples))
        src_id, tgt_id, year = pos_samples[seed_idx]

        # safety: ensure the positive's year is within the window (else resample from available_years)
        if year < _STUDY_YEAR_MIN or year > _STUDY_YEAR_MAX:
            year = available_years[rng.randint(0, n_years)]

        r = rng.random()

        if r < 0.2:
            # Intra-industry target replacement
            category = company2industry.get(src_id)
            if category and category in ind2comp_lists:
                candidates = ind2comp_lists[category]
                new_tgt = candidates[rng.randint(0, len(candidates))]
            else:
                new_tgt = company_ids[rng.randint(0, n_companies)]
            triple = (src_id, new_tgt, year)
        elif r < 0.5:
            # Intra-industry source replacement
            category = company2industry.get(src_id)
            if category and category in ind2comp_lists:
                candidates = ind2comp_lists[category]
                new_src = candidates[rng.randint(0, len(candidates))]
            else:
                new_src = company_ids[rng.randint(0, n_companies)]
            triple = (new_src, tgt_id, year)
        elif r < 0.7:
            # Random target replacement
            new_tgt = company_ids[rng.randint(0, n_companies)]
            triple = (src_id, new_tgt, year)
        elif r < 1.0:
            # Random source replacement
            new_src = company_ids[rng.randint(0, n_companies)]
            triple = (new_src, tgt_id, year)
        else:
            # Fallback: different year
            new_year = available_years[rng.randint(0, n_years)]
            if new_year == year:
                new_year = available_years[(available_years.index(year) + 1) % n_years]
            triple = (src_id, tgt_id, new_year)

        if triple not in blacklist and triple not in negatives:
            negatives.append(triple)
        attempts += 1

    return negatives


def build_bootstrap_static_graph(bootstrap_data: dict, iteration_dataset: BootstrapIterationDataset):
    """Build the static graph for bootstrap training: node features from full_dataset, edges from train_pool (compatible with the frozen backbone)."""
    full_dataset = bootstrap_data['full_dataset']
    reverse_node_mapping = full_dataset.reverse_node_mapping
    company_embeddings = full_dataset.company_embeddings

    node_features = [
        company_embeddings[reverse_node_mapping[i]]
        for i in range(len(reverse_node_mapping))
    ]

    train_pool = bootstrap_data['train_pool']
    edge_indices = []
    edge_times = []
    node_mapping = full_dataset.node_mapping

    for source_id, target_id, year in train_pool:
        source_idx = node_mapping[source_id]
        target_idx = node_mapping[target_id]
        edge_indices.append([source_idx, target_idx])
        edge_times.append(year)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    edge_time = torch.tensor(edge_times, dtype=torch.float)

    dynamic_data = Data(
        x=x,
        edge_index=edge_index,
        edge_time=edge_time,
        num_nodes=len(node_features)
    )

    return dynamic_data


def create_dataloaders(negative_ratio=1, embedding_name="embedding",
                       batch_size=32, train_ratio=0.8, toy_mode=False,
                       random_state=None, min_degree=2, source_filter='semi',
                       other_possible_fill=0.0, filter_factset_neg=False,
                       intra_industry_neg=True, use_pred_neg=True):
    """Create train/val/test data loaders, returning (static_data, train_loader, val_loader, test_loader, full_dataset)."""
    bootstrap_data = create_bootstrap_datasets(
        negative_ratio=negative_ratio,
        embedding_name=embedding_name,
        test_ratio=0.10,
        val_ratio=0.10,
        random_state=random_state if random_state is not None else 42,
        min_degree=min_degree,
        source_filter=source_filter,
        other_possible_fill=other_possible_fill,
        filter_factset_neg=filter_factset_neg,
        intra_industry_neg=intra_industry_neg,
        toy_mode=toy_mode,
    )

    # Standard training: use all of train_pool, no bootstrap resampling
    train_dataset = BootstrapIterationDataset(
        train_pool=bootstrap_data['train_pool'],
        blacklist=bootstrap_data['blacklist'],
        meta_dataset=bootstrap_data['train_pool_dataset'],
        negative_ratio=negative_ratio,
        seed=random_state if random_state is not None else 42,
        intra_industry_neg=intra_industry_neg,
        bootstrap=False,
    )

    # num_workers is unavailable on Windows, so leave it empty
    train_kwargs = {} if sys.platform == 'win32' else {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    test_kwargs = {} if sys.platform == 'win32' else {'num_workers': 2, 'pin_memory': True}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **train_kwargs,
    )

    val_loader = DataLoader(
        bootstrap_data['val_dataset'],
        batch_size=batch_size,
        shuffle=False,
        **test_kwargs,
    )

    test_loader = DataLoader(
        bootstrap_data['test_dataset'],
        batch_size=batch_size,
        shuffle=False,
        **test_kwargs,
    )

    static_data = build_bootstrap_static_graph(bootstrap_data, train_dataset)

    return static_data, train_loader, val_loader, test_loader, bootstrap_data['full_dataset']


def load_pretrained_backbone(model, checkpoint_path: str, device: str = 'cpu'):
    """Load weights from a pretrained checkpoint and freeze the backbone (static_encoder or static_gnn), returning model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint)

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[Pretrained] missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[Pretrained] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    if hasattr(model, 'static_encoder'):
        _freeze_module(model.static_encoder)
        print("[Pretrained] static_encoder frozen (requires_grad=False)")
    elif hasattr(model, 'static_gnn'):
        _freeze_module(model.static_gnn)
        print("[Pretrained] static_gnn frozen (requires_grad=False)")
    else:
        print("[Pretrained] WARNING: no freezable backbone module found")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Pretrained] trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model


def _freeze_module(module):
    """Freeze all parameters of a module"""
    for param in module.parameters():
        param.requires_grad = False


def reinit_trainable_parts(model):
    """Reinitialize the trainable parts (temporal_encoder+edge_predictor of GAT-GRU/LSTM/TNA, or dynamic_gnn+link_predictor of SEAL), keeping the backbone frozen."""
    def _reset_lstm_gru(m):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.zero_()
                n = param.data.size(0)
                start, end = n // 3, n // 3 * 2
                param.data[start:end].fill_(1.0)

    def _reset_linear(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        elif hasattr(m, 'weight') and m.weight.dim() >= 1:
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

    def _reset_embedding(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def _reset_gcnconv(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    # SEAL model branch: dynamic_gnn + link_predictor
    if hasattr(model, 'dynamic_gnn') and hasattr(model, 'static_gnn'):
        for module in model.dynamic_gnn.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)
            elif isinstance(module, nn.Embedding):
                _reset_embedding(module)
        for p in model.dynamic_gnn.parameters():
            p.requires_grad = True
        print("  [Reinit] dynamic_gnn reinitialized")

        if hasattr(model, 'link_predictor'):
            for module in model.link_predictor.modules():
                if isinstance(module, nn.Linear):
                    _reset_linear(module)
            for p in model.link_predictor.parameters():
                p.requires_grad = True
            print("  [Reinit] link_predictor reinitialized")

        # Confirm static_gnn is still frozen
        for p in model.static_gnn.parameters():
            if p.requires_grad:
                p.requires_grad = False
                print("  [WARNING] unfrozen parameter found in static_gnn, forcibly frozen")
        return

    # GAT-GRU/LSTM/TNA model branch
    if hasattr(model, 'temporal_encoder'):
        _reset_lstm_gru(model.temporal_encoder)
        for p in model.temporal_encoder.parameters():
            p.requires_grad = True
        print("  [Reinit] temporal_encoder reinitialized")

    if hasattr(model, 'edge_predictor'):
        for module in model.edge_predictor.modules():
            if isinstance(module, nn.Linear):
                _reset_linear(module)
        for p in model.edge_predictor.parameters():
            p.requires_grad = True
        print("  [Reinit] edge_predictor reinitialized")

    # Confirm static_encoder is still frozen
    if hasattr(model, 'static_encoder'):
        for p in model.static_encoder.parameters():
            if p.requires_grad:
                p.requires_grad = False
                print("  [WARNING] unfrozen parameter found in static_encoder, forcibly frozen")
