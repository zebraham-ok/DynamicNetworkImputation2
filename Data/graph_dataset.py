"""
Low-level dataset primitives (helper module): CompanySupplyDataset (link-prediction dataset) and build_static_graph (static graph construction).

The public entry point is `Data.company_dataset`, which re-exports the names defined here.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Data.neo4j_SPLC import Neo4jClient
from tqdm import tqdm
import logging
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

import torch
from torch.utils.data import Dataset
import random
import numpy as np
from typing import List, Tuple, Dict
from torch_geometric.data import Data

# Default negative-sample file path (relative to this module's directory)
_DEFAULT_NEG_DIR = os.path.join(os.path.dirname(__file__), "stopped_neg_sample.csv")


def load_predefined_negatives(neg_dir, company_ids):
    """Load the predefined negatives (CSV) and keep only triples whose endpoints exist in the graph.

    Single source of truth for parsing the file: `CompanySupplyDataset.__init__` and the three-way
    split in `Data.company_dataset` both call this function, so the CSV can never be interpreted in
    two different ways. Duplicate triples are collapsed (first occurrence kept, file order
    preserved) so the result is deterministic.
    """
    with open(neg_dir, encoding='utf-8') as f:
        s = f.read().strip()
    str_list = s.split("\n")
    # Compatible with old and new formats (take first/second/last column)
    triples = [(int(parts[0]), int(parts[1]), int(parts[-1]))
               for i in str_list[1:] if (parts := i.split(","))]
    ids = set(company_ids)
    triples = [t for t in triples if t[0] in ids and t[1] in ids]

    seen = set()
    unique = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


class CompanySupplyDataset(Dataset):
    """Link-prediction dataset class.

    When toy_mode=True it contains only a small number of samples (<128), for quick testing.
    """

    def __init__(self, negative_ratio: int = 1, embedding_name="embedding", 
                 neg_dir=_DEFAULT_NEG_DIR,
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 toy_mode=False, sample_seed=None, neg_only=False,
                 positive_samples=None, predifined_neg_data=None, fixed_neg_data=None,
                 min_degree=2, source_filter='semi', other_possible_fill=0.0,
                 filter_factset_neg=False, intra_industry_neg=True, use_pred_neg=True,
                 use_attr_onehot=False, onehot_attrs=None):
        """
        Args:
            min_degree: both endpoint nodes of an edge must have degree >= min_degree (0=no filter)
            source_filter: filter on the Supply edge's source attribute (None=no filter, 'semi'=use semi-structured data only)
            intra_industry_neg: whether to use intra-industry negative sampling (True=industry-aware, False=purely random)
            predifined_neg_data: predefined negatives (in the study pipeline: the CSV share of the
                three-way split of stopped_neg_sample.csv). Counted only when ``use_pred_neg`` is True.
            fixed_neg_data: negatives belonging to the frozen split itself (val/test fixed negatives plus
                the CSV share assigned to train). Used unconditionally - NOT controlled by ``use_pred_neg``.
            use_pred_neg: whether to use the CSV predefined negatives read from ``neg_dir``
                (stopped_neg_sample.csv). When False the CSV is not used and any ``predifined_neg_data``
                is ignored; ``fixed_neg_data`` is unaffected.
            use_attr_onehot: if True, append one-hot blocks of the categorical node attributes listed in
                ``onehot_attrs`` to the node embedding, so that X = [embedding | one-hot(...)]. This is the
                switch that makes the shared linear feature extractor (Models.common.NodeFeatureExtractor)
                an actual dimensionality reduction instead of a 128 -> 128 re-encoder.
            onehot_attrs: Neo4j property names to one-hot encode, e.g. ['country', 'industry_2nd', 'category_3rd'].
                The vocabulary is built from the data at load time (sorted for reproducibility), so no
                vocabulary file has to be shipped with the dataset. An explicit ``<NA>`` category is
                reserved at index 0 so that nodes with a missing attribute are still distinguishable.
        """
        self.embedding_name = embedding_name
        self.negative_ratio = negative_ratio
        self.neo4j_host = Neo4jClient()
        self.device=device
        self.toy_mode=toy_mode
        self.neg_only=neg_only
        self.min_degree = min_degree
        self.source_filter = source_filter
        self.other_possible_fill = other_possible_fill
        self.filter_factset_neg = filter_factset_neg
        self.intra_industry_neg = intra_industry_neg
        self.use_pred_neg = use_pred_neg
        self.use_attr_onehot = bool(use_attr_onehot)
        self.onehot_attrs = list(onehot_attrs) if onehot_attrs else []
        if self.use_attr_onehot and not self.onehot_attrs:
            raise ValueError(
                "use_attr_onehot=True requires onehot_attrs, e.g. "
                "onehot_attrs: ['country', 'industry_2nd', 'category_3rd']"
            )
        # Filled in by _cache_company_info (None when the one-hot option is disabled)
        self.attr_onehot_matrix = None
        self.attr_onehot_row = {}
        self.attr_onehot_vocabs = {}
        self.attr_onehot_sizes = {}
        self.attr_onehot_dim = 0
        
        if isinstance(sample_seed, int):
            self.random=random.Random(sample_seed)
        else:
            self.random=random.Random(42)
        
        self._cache_company_info()

        # factset edges are used for external validation metrics, not training
        self.factset_edges = self._get_factset_edges()

        if positive_samples is not None:
            self.original_positive_samples = positive_samples
        else:
            self.original_positive_samples = self._get_positive_samples()

        # Positive-edge cache, replacing _check_relation_exists's Neo4j query (speeds up negative sampling)
        self._known_edges_cache = set(self.original_positive_samples)
        if self.filter_factset_neg:
            self._known_edges_cache.update(self.factset_edges)

        if positive_samples is None:
            self._build_year_distribution()
            self.positive_samples = self._resample_positive_samples()
        else:
            self.positive_samples = self.original_positive_samples

        if predifined_neg_data is not None:
            self.predifined_neg_data = list(predifined_neg_data)
        else:
            self.predifined_neg_data = []
            if neg_dir:
                self.predifined_neg_data = load_predefined_negatives(neg_dir, self.company_ids)
                print(f"[Dataset] Predefined negatives (CSV): {len(self.predifined_neg_data)} unique triples "
                      f"from {os.path.basename(neg_dir)} (counted only if use_pred_neg={self.use_pred_neg})")

        # Negatives that belong to the frozen split itself: the val/test fixed negatives, plus the CSV
        # share that the three-way split assigns to train. These are always used, no matter what
        # `use_pred_neg` says - that flag describes the CSV, not the split.
        self.fixed_neg_data = list(fixed_neg_data) if fixed_neg_data is not None else []
    
    def _cache_company_info(self):
        """Cache company node info to avoid repeated queries"""
        # Optional categorical attributes, fetched with neutral aliases (attr_0, attr_1, ...) so that
        # any property name - including ones clashing with the aliases used below - can be requested.
        attr_select = "".join(
            f", c.`{name}` as attr_{i}" for i, name in enumerate(self.onehot_attrs)
        ) if self.use_attr_onehot else ""

        result = self.neo4j_host.execute_query(f"""
            MATCH (c:EntityObj)
            WHERE c.{self.embedding_name} IS NOT NULL
            RETURN id(c) as company_id, c.{self.embedding_name} as embedding, c.industry_1st as industry{attr_select}
        """)
        
        # neo4j company_id → PyG node index
        self.node_mapping = {record["company_id"]:i for i, record in enumerate(result)}
        self.reverse_node_mapping = {v: k for k, v in self.node_mapping.items()}
        
        self.company_embeddings = {}
        self.company_ids = []
        self.industry2company = {}
        self.company2industry = {}
        
        for record in result:
            company_id = record["company_id"]
            embedding = record["embedding"]
            industry = record["industry"]
            if embedding is not None:
                if self.other_possible_fill != 0.0:
                    embedding_arr = np.array(embedding, dtype=np.float32)
                    embedding_arr[embedding_arr == 0] = self.other_possible_fill
                    embedding = embedding_arr.tolist()
                self.company_embeddings[company_id] = embedding
                self.company_ids.append(company_id)
                
                self.company2industry[company_id] = industry
                if industry not in self.industry2company:
                    self.industry2company[industry] = [company_id]
                else:
                    self.industry2company[industry].append(company_id)

        if self.use_attr_onehot:
            self._build_attr_onehot(result)

        # All years (restricted to the 2013-2025 study window)
        result = self.neo4j_host.execute_query("""
            MATCH ()-[r:SupplyProductTo]->()
            WHERE r.year IS NOT NULL AND r.year >= 2013 AND r.year <= 2025
            RETURN DISTINCT r.year as year
            ORDER BY r.year
        """)
        self.available_years = [record["year"] for record in result]

    def _build_attr_onehot(self, result):
        """Build the optional one-hot attribute blocks that are appended to the node embedding.

        The vocabulary is derived from the data itself and sorted, so the encoding is
        reproducible without shipping a vocabulary file.  Index 0 of every block is reserved
        for a missing / empty value (``<NA>``), therefore each node has exactly one active
        column per block and "attribute missing" is distinguishable from "attribute = other".
        """
        missing_token = '<NA>'
        attr_list = self.onehot_attrs
        raw_values = [[] for _ in attr_list]
        seen_ids = []

        for record in result:
            company_id = record["company_id"]
            if record["embedding"] is None or company_id in seen_ids:
                continue
            seen_ids.append(company_id)
            for i in range(len(attr_list)):
                raw_values[i].append(record.get(f"attr_{i}"))

        blocks = []
        offset = 0
        for name, values in zip(attr_list, raw_values):
            normalised = [missing_token if v is None or v == '' else str(v) for v in values]
            categories = sorted({v for v in normalised if v != missing_token})
            categories = [missing_token] + categories
            index_of = {c: j for j, c in enumerate(categories)}

            if len(categories) <= 1:
                raise ValueError(
                    f"onehot_attrs entry '{name}' has no value at all (all {len(normalised)} nodes are "
                    f"NULL/empty). Check that the property name matches the Neo4j schema "
                    f"(anonymised releases may rename it)."
                )

            block = np.zeros((len(normalised), len(categories)), dtype=np.float32)
            for row, value in enumerate(normalised):
                block[row, index_of[value]] = 1.0

            blocks.append(block)
            self.attr_onehot_vocabs[name] = categories
            self.attr_onehot_sizes[name] = len(categories)
            offset += len(categories)

        self.attr_onehot_matrix = np.concatenate(blocks, axis=1) if blocks else None
        self.attr_onehot_dim = offset
        self.attr_onehot_row = {company_id: i for i, company_id in enumerate(seen_ids)}

    def describe_node_features(self, feature_dim: int = None) -> str:
        """Human-readable description of how X is assembled (printed once per graph build)."""
        embedding_dim = len(next(iter(self.company_embeddings.values()))) if self.company_embeddings else 0
        if self.attr_onehot_matrix is None:
            text = (f"[Features] X = embedding({embedding_dim}) = {feature_dim if feature_dim else embedding_dim} "
                    f"(attribute one-hot DISABLED)")
            if self.onehot_attrs:
                text += f"; enable it with dataset.use_attr_onehot=true (attrs: {self.onehot_attrs})"
            return text
        blocks = ", ".join(f"{name}={self.attr_onehot_sizes[name]}" for name in self.onehot_attrs)
        total = embedding_dim + self.attr_onehot_dim
        return (f"[Features] X = embedding({embedding_dim}) + one-hot[{blocks}] "
                f"= {total} (attribute one-hot ENABLED)")

    def _build_year_distribution(self):
        """Build year-distribution info for uniform sampling (align to the year with the fewest samples, avoiding year bias and leakage)"""
        self.year_counts = {}
        for source_id, target_id, year in self.original_positive_samples:
            self.year_counts[year] = self.year_counts.get(year, 0) + 1

        self.target_samples_per_year = min(self.year_counts.values())
        self.year_weights = {}
        total_samples = len(self.original_positive_samples)
        
        for year, count in self.year_counts.items():
            self.year_weights[year] = total_samples / count
        
        total_weight = sum(self.year_weights.values())
        for year in self.year_weights:
            self.year_weights[year] /= total_weight
        
        self.weighted_years = []
        for year, weight in self.year_weights.items():
            self.weighted_years.extend([year] * int(weight * 1000))
    
    def _resample_positive_samples(self):
        """Resample the positives so that each year has an equal number of samples"""
        samples_by_year = {}
        for source_id, target_id, year in self.original_positive_samples:
            if year not in samples_by_year:
                samples_by_year[year] = []
            samples_by_year[year].append((source_id, target_id, year))
        
        resampled_samples = []
        for year, samples in samples_by_year.items():
            if len(samples) < self.target_samples_per_year:
                selected = self.random.choices(samples, k=self.target_samples_per_year)
            else:
                selected = self.random.sample(samples, self.target_samples_per_year)
            resampled_samples.extend(selected)
        random.shuffle(resampled_samples)
        return resampled_samples
    
    def _generate_negative_sample(self, positive_triple: Tuple) -> Tuple:
        """Generate a negative for a given positive, using uniform year sampling
        
        Controlled by self.intra_industry_neg to enable/disable the intra-industry replacement strategy.
        """
        source_id, target_id, year = positive_triple
        
        if self.intra_industry_neg:
            return self._generate_negative_with_industry(source_id, target_id, year)
        else:
            return self._generate_negative_random(source_id, target_id, year)
    
    def _generate_negative_with_industry(self, source_id, target_id, year):
        """Intra-industry negative sampling"""
        random_number = random.random()

        # Intra-industry target-company replacement (20%)
        if random_number < 0.2:
            category = self.company2industry.get(source_id)
            if category in self.industry2company:
                for _ in range(10):
                    new_target = random.choice(self.industry2company[category])
                    if new_target != target_id and not self._check_relation_exists(source_id, new_target, year):
                        return (source_id, new_target, year)
            # Fall back to random selection on failure
            new_target = random.choice(self.company_ids)
            while self._check_relation_exists(source_id, new_target, year):
                new_target = random.choice(self.company_ids)
            return (source_id, new_target, year)
        
        # Intra-industry source-company replacement (30%)
        elif random_number < 0.5:
            category = self.company2industry.get(source_id)
            if category in self.industry2company:
                for _ in range(10):
                    new_source = random.choice(self.industry2company[category])
                    if new_source != source_id and not self._check_relation_exists(new_source, target_id, year):
                        return (new_source, target_id, year)
            # Fall back to random selection on failure
            new_source = random.choice(self.company_ids)
            while self._check_relation_exists(new_source, target_id, year):
                new_source = random.choice(self.company_ids)
            return (new_source, target_id, year)

        # Random target-company replacement (20%)
        elif random_number < 0.7:
            new_target = random.choice(self.company_ids)
            while self._check_relation_exists(source_id, new_target, year):
                new_target = random.choice(self.company_ids)
            return (source_id, new_target, year)

        # Random source-company replacement (30%)
        elif random_number < 1:
            new_source = random.choice(self.company_ids)
            while self._check_relation_exists(new_source, target_id, year):
                new_source = random.choice(self.company_ids)
            return (new_source, target_id, year)

        # Keep the company pair fixed and use a different year (fallback)
        else:
            return self._generate_negative_time_swap(source_id, target_id, year)
    
    def _generate_negative_random(self, source_id, target_id, year):
        """Purely random negative sampling (no intra-industry info)"""
        # 50% target replacement, 50% source replacement
        if random.random() < 0.5:
            new_target = random.choice(self.company_ids)
            while self._check_relation_exists(source_id, new_target, year):
                new_target = random.choice(self.company_ids)
            return (source_id, new_target, year)
        else:
            new_source = random.choice(self.company_ids)
            while self._check_relation_exists(new_source, target_id, year):
                new_source = random.choice(self.company_ids)
            return (new_source, target_id, year)
    
    def _generate_negative_time_swap(self, source_id, target_id, year):
        """Keep the company pair fixed and use a different year"""
        for _ in range(10):
            sampled_year = random.choice(self.weighted_years)
            if sampled_year != year and not self._check_relation_exists(source_id, target_id, sampled_year):
                return (source_id, target_id, sampled_year)
        # Fallback: random replacement
        new_target = random.choice(self.company_ids)
        sampled_year = random.choice(self.weighted_years)
        while self._check_relation_exists(source_id, new_target, sampled_year):
            new_target = random.choice(self.company_ids)
        return (source_id, new_target, sampled_year)
    
    def _get_positive_samples(self, limit: int = None) -> List[Tuple]:
        """Fetch positives, supporting source filtering and degree filtering"""
        source_clause = ""
        if self.source_filter:
            source_clause = f" AND r.source = '{self.source_filter}'"
        
        query = f"""
            MATCH (c1:EntityObj)-[r:SupplyProductTo]->(c2:EntityObj)
            WHERE c1.{self.embedding_name} IS NOT NULL AND c2.{self.embedding_name} IS NOT NULL
            AND r.year IS NOT NULL AND r.year >= 2013 AND r.year <= 2025{source_clause}
            RETURN id(c1) as source_id, id(c2) as target_id, r.year as year
        """
        if limit:
            query += f" LIMIT {limit}"
            
        result = self.neo4j_host.execute_query(query)
        samples = [(record["source_id"], record["target_id"], record["year"]) 
                  for record in result]

        if self.min_degree > 0:
            degree = {}
            for src, tgt, yr in samples:
                degree[src] = degree.get(src, 0) + 1
                degree[tgt] = degree.get(tgt, 0) + 1
            samples = [(s, t, y) for s, t, y in samples 
                       if degree.get(s, 0) >= self.min_degree and degree.get(t, 0) >= self.min_degree]
        
        return samples

    def _get_factset_edges(self) -> List[Tuple]:
        """Fetch edges present in factset but not in semi (used for external validation metrics).

        Retention conditions: both endpoints have an embedding and appear in semi, the triple does not
        overlap with semi, and the year lies within semi's year range (to avoid scoring unseen years).
        """
        # factset edges (restricted to the 2013-2025 study window)
        query = f"""
            MATCH (c1:EntityObj)-[r:SupplyProductTo]->(c2:EntityObj)
            WHERE c1.{self.embedding_name} IS NOT NULL AND c2.{self.embedding_name} IS NOT NULL
            AND r.year IS NOT NULL AND r.year >= 2013 AND r.year <= 2025 AND r.source = 'factset'
            RETURN id(c1) as source_id, id(c2) as target_id, r.year as year
        """
        factset_result = self.neo4j_host.execute_query(query)
        factset_raw = set()
        for record in factset_result:
            sid = record["source_id"]
            tid = record["target_id"]
            yr = record["year"]
            if sid in self.node_mapping and tid in self.node_mapping:
                factset_raw.add((sid, tid, yr))
        print(f"[Factset] Raw factset edges (with embedding and in node_mapping): {len(factset_raw)}")

        # semi edge set (used for dedup and for extracting the node set)
        if self.source_filter == 'semi':
            semi_set = set((s, t, y) for s, t, y in self._get_positive_samples())
        else:
            semi_query = f"""
                MATCH (c1:EntityObj)-[r:SupplyProductTo]->(c2:EntityObj)
                WHERE c1.{self.embedding_name} IS NOT NULL AND c2.{self.embedding_name} IS NOT NULL
                AND r.year IS NOT NULL AND r.year >= 2013 AND r.year <= 2025 AND r.source = 'semi'
                RETURN id(c1) as source_id, id(c2) as target_id, r.year as year
            """
            semi_result = self.neo4j_host.execute_query(semi_query)
            semi_set = set((r["source_id"], r["target_id"], r["year"]) for r in semi_result)
        
        semi_nodes = set()
        for s, t, y in semi_set:
            semi_nodes.add(s)
            semi_nodes.add(t)
        print(f"[Factset] semi edges: {len(semi_set)}, semi nodes involved: {len(semi_nodes)}")

        factset_no_overlap = factset_raw - semi_set

        # Keep only edges whose both endpoints appeared in semi
        factset_only = [(s, t, y) for s, t, y in factset_no_overlap
                         if s in semi_nodes and t in semi_nodes]

        # Keep only edges whose year is within semi's data range (consistent with train/test year range)
        semi_years = set(y for _, _, y in semi_set)
        factset_only = [(s, t, y) for s, t, y in factset_only if y in semi_years]
        print(f"[Factset] Final edges used for external validation after filtering: {len(factset_only)} (semi years: {sorted(semi_years)})")

        return factset_only

    def _check_relation_exists(self, source_id: int, target_id: int, year: int):
        """Check whether a relation exists (in-memory cache version)"""
        return (source_id, target_id, year) in self._known_edges_cache
    
    def __len__(self):
        """Dynamically compute the dataset size"""
        pred_neg_count = len(self.predifined_neg_data) if self.use_pred_neg else 0
        # `fixed_neg_data` is part of the split protocol, `predifined_neg_data` is the CSV share
        assumed_len = (len(self.positive_samples) * (1 + self.negative_ratio)
                       + len(self.fixed_neg_data) + pred_neg_count)
        if self.toy_mode and self.neg_only:
            if len(self.positive_samples)*self.negative_ratio>128:
                return 128
            else:
                return len(self.positive_samples)*self.negative_ratio
        elif self.neg_only:
            return len(self.positive_samples)*self.negative_ratio
        elif self.toy_mode and assumed_len>128:
            return 128
        else:
            return assumed_len 
    
    def __getitem__(self, idx):
        """Fetch a training sample in the format required by the new model"""
        total_positive = len(self.positive_samples)
        dynamic_neg_count = total_positive * self.negative_ratio
        
        if self.neg_only:
            positive_idx = (idx - total_positive) % total_positive
            source_id, target_id, year = self._generate_negative_sample(
                self.positive_samples[positive_idx]
            )
            label = 0.0
        else:
            if idx < total_positive:
                # Positive sample
                source_id, target_id, year = self.positive_samples[idx]
                label = 1.0
            elif idx < total_positive + dynamic_neg_count:
                # Dynamically generated negative sample
                positive_idx = (idx - total_positive) % total_positive
                source_id, target_id, year = self._generate_negative_sample(
                    self.positive_samples[positive_idx]
                )
                label = 0.0
            elif idx < total_positive + dynamic_neg_count + len(self.fixed_neg_data):
                # Fixed negative sample of the frozen split (val/test), or of the CSV share that the
                # three-way split assigned to train
                negative_idx = idx - total_positive - dynamic_neg_count
                source_id, target_id, year = self.fixed_neg_data[negative_idx]
                label = 0.0
            elif self.use_pred_neg:
                # Predefined negative sample (CSV)
                negative_idx = idx - total_positive - dynamic_neg_count - len(self.fixed_neg_data)
                source_id, target_id, year = self.predifined_neg_data[negative_idx]
                label = 0.0
            else:
                # Fallback (this branch should not occur when use_pred_neg=False): return a dynamic negative
                positive_idx = idx % total_positive
                source_id, target_id, year = self._generate_negative_sample(
                    self.positive_samples[positive_idx]
                )
                label = 0.0

        # Node indices are consistent across train/test/full (mapping provided by full_dataset)
        source_idx = self.node_mapping[source_id]
        target_idx = self.node_mapping[target_id]

        # Output stays on CPU; the training loop is responsible for .to(device)
        link_indices = torch.tensor([source_idx, target_idx], dtype=torch.long)
        current_time = torch.tensor(year, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return link_indices, current_time, label

def assemble_node_features(full_set: CompanySupplyDataset) -> torch.Tensor:
    """Assemble the node feature matrix X used by every backbone model.

    X = [ dense text embedding | optional one-hot attribute blocks ] with the row order given by
    ``full_set.reverse_node_mapping`` (i.e. PyG node index -> neo4j id).  This is the single place
    where node features are materialised, so training and bootstrap reuse exactly the same layout.
    """
    reverse_node_mapping = full_set.reverse_node_mapping
    company_embeddings = full_set.company_embeddings
    # company_embeddings is indexed by neo4j_index, so it must be inverse-mapped via reverse_node_mapping
    node_features = [company_embeddings[reverse_node_mapping[i]] for i in range(len(reverse_node_mapping))]
    features = np.asarray(node_features, dtype=np.float32)

    matrix = getattr(full_set, 'attr_onehot_matrix', None)
    if matrix is not None and matrix.size > 0:
        row_of = getattr(full_set, 'attr_onehot_row', {})
        extra = np.stack([matrix[row_of[reverse_node_mapping[i]]] for i in range(len(reverse_node_mapping))])
        features = np.concatenate([features, extra], axis=1)

    return torch.tensor(features, dtype=torch.float)


def build_static_graph(full_set:CompanySupplyDataset, edge_info_dataset: Dataset):
    """Build the static graph data object: node info from full_set, edge info from edge_info_dataset (the two may be the same or different)."""

    node_features = assemble_node_features(full_set)
    print(getattr(full_set, 'describe_node_features', lambda: '')())

    edge_indices=[]
    edge_times=[]
    # Iterate the positives directly to avoid triggering negative generation (dynamic negative sampling calls Neo4j queries)
    if hasattr(edge_info_dataset, 'positive_samples') and edge_info_dataset.positive_samples is not None:
        for source_id, target_id, year in tqdm(edge_info_dataset.positive_samples, desc="loading static graph"):
            source_idx = edge_info_dataset.node_mapping[source_id]
            target_idx = edge_info_dataset.node_mapping[target_id]
            edge_indices.append([source_idx, target_idx])
            edge_times.append(year)
    else:
        for link_indices, current_time, label in tqdm(edge_info_dataset, desc="loading static graph"):
            if label:
                edge_indices.append(link_indices.tolist())
                edge_times.append(current_time.item())
    
    # Convert to a PyG Data object
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = node_features
    edge_time = torch.tensor(edge_times, dtype=torch.float)
    
    dynamic_data = Data(
        x=x,
        edge_index=edge_index,
        edge_time=edge_time,
        num_nodes=x.size(0)
    )
    
    return dynamic_data
