import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
import networkx as nx
from collections import deque

from Models.common import build_feature_extractor, apply_node_feature_extractor

class BatchSubgraphExtractor:
    """Batch subgraph extraction"""
    def __init__(self, k_hop=2, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.k_hop = k_hop
        self.batch_size = batch_size
        self.device=device
        
    def batch_extract_subgraphs(self, data, link_indices, current_times):
        """link_indices: [B, 2] link node indices; current_times: [B] current times. Returns a list of batch subgraphs."""
        batch_subgraphs = []
        
        # Extract the subgraph for all nodes involved in one batch together (speeds up computation)
        for i in range(0, len(link_indices), self.batch_size):
            batch_links = link_indices[i:i+self.batch_size]
            batch_times = current_times[i:i+self.batch_size]
            
            batch_data = self._extract_batch_subgraphs(data, batch_links, batch_times)
            batch_subgraphs.append(batch_data)
            
        return batch_subgraphs
    
    def _extract_batch_subgraphs(self, data, batch_links, batch_times):
        """Extract the subgraphs of a single batch"""
        subgraph_list = []
        
        for (a_idx, b_idx), current_time in zip(batch_links, batch_times):
            subgraph_data = self._extract_single_subgraph(data, a_idx, b_idx, current_time)
            # Convert to a PyG Data object for convenient batch processing
            pyg_data = Data(
                x=subgraph_data['x'],
                edge_index=subgraph_data['edge_index'],
                edge_attr=subgraph_data['temporal_weights'].unsqueeze(1),
                node_labels=subgraph_data['node_labels'],
                a_mapping=torch.tensor([subgraph_data['a_mapping']]),
                b_mapping=torch.tensor([subgraph_data['b_mapping']]),
                num_nodes=subgraph_data['x'].size(0)
            ).to(self.device)
            subgraph_list.append(pyg_data)
        
        # Use PyG's Batch to merge the subgraphs
        batch = Batch.from_data_list(subgraph_list)
        return batch
    
    def _extract_single_subgraph(self, data, a_idx, b_idx, current_time):
        """Extract the k-hop neighborhood subgraph containing nodes A and B, and label its nodes.
        k_hop_subgraph(relabel_nodes=True) renumbers the nodes within the subgraph to 0..S-1."""
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            [a_idx, b_idx], self.k_hop, data.edge_index, relabel_nodes=True, 
            num_nodes=data.num_nodes
        )
        
        # Fetch the node features within the subgraph
        subgraph_x = data.x[subset]
        
        # Fetch the time information of the edges within the subgraph
        subgraph_edge_time = data.edge_time[edge_mask]
        
        # Compute the temporal weights
        time_diff = torch.abs(subgraph_edge_time - current_time)
        temporal_weights = 1.0 / (time_diff + 1.0)
        
        # Look up the relabeled positions of A and B within subset
        a_mask = (subset == a_idx)
        b_mask = (subset == b_idx)
        
        # Fault tolerance: if a node is not in the subgraph (rare edge case), fall back to 0
        a_mapping_val = a_mask.nonzero(as_tuple=True)[0][0].item() if a_mask.any() else 0
        b_mapping_val = b_mask.nonzero(as_tuple=True)[0][0].item() if b_mask.any() else 0
        
        # Node labels: based on shortest-path distance to A and B (subgraph-local BFS)
        node_labels = self._create_node_labels(
            edge_index, subset, a_idx, b_idx
        )
        
        subgraph_data = {
            'x': subgraph_x,
            'edge_index': edge_index,
            'edge_time': subgraph_edge_time,
            'temporal_weights': temporal_weights,
            'node_labels': node_labels,
            'a_mapping': a_mapping_val,
            'b_mapping': b_mapping_val,
            'original_nodes': subset  # node indices in the original graph
        }
        
        return subgraph_data
    
    def _create_node_labels(self, subgraph_edge_index, subset, a_idx_orig, b_idx_orig):
        """Compute DRNL (Double-Radius Node Labeling) based on a subgraph-local BFS.

The BFS is performed inside the subgraph (k_hop_subgraph has already extracted all nodes
within 2 hops, so the intra-subgraph distances equal the true distances in the full graph).
        """
        S = len(subset)
        
        # Build a NetworkX graph locally on the subgraph (number of nodes = S)
        G_sub = nx.Graph()
        G_sub.add_nodes_from(range(S))
        if subgraph_edge_index.numel() > 0:
            G_sub.add_edges_from(subgraph_edge_index.t().tolist())
        
        a_pos = (subset == a_idx_orig).nonzero(as_tuple=True)[0]
        b_pos = (subset == b_idx_orig).nonzero(as_tuple=True)[0]
        
        # BFS from A (cutoff=k_hop*2+1, beyond which nodes are unreachable)
        dist_a = {}
        if len(a_pos) > 0:
            dist_a = nx.single_source_shortest_path_length(
                G_sub, a_pos[0].item(), cutoff=self.k_hop * 2 + 1
            )
        
        # BFS from B
        dist_b = {}
        if len(b_pos) > 0 and b_pos[0].item() < S:
            dist_b = nx.single_source_shortest_path_length(
                G_sub, b_pos[0].item(), cutoff=self.k_hop * 2 + 1
            )
        
        labels = []
        for node in range(S):
            d_a = dist_a.get(node, float('inf'))
            d_b = dist_b.get(node, float('inf'))
            labels.append(self._encode_distance_pair(d_a, d_b))
        
        return torch.tensor(labels, dtype=torch.long)

    def _encode_distance_pair(self, dist_a, dist_b):
        """Encode the distance pair (distance_to_a, distance_to_b) into a single integer using SEAL's Double-Radius Node Labeling (DRNL)."""
        if dist_a == float('inf') or dist_b == float('inf'):
            return 0  # unreachable nodes are labeled 0
            
        # DRNL encoding scheme
        d1 = min(dist_a, dist_b)
        d2 = max(dist_a, dist_b)
        
        # Map (d1, d2) into a single-integer space
        label = 1 + d1 + (d2 * (d2 + 1)) // 2
        
        max_label_value = self.k_hop*2+1
        return min(label, max_label_value)

    def _bfs_shortest_path(self, graph, start_node, target_nodes):
        """Compute the shortest-path distances from a start node to a set of target nodes using BFS."""
        distances = {}
        visited = set()
        queue = deque([(start_node, 0)])
        
        while queue and len(distances) < len(target_nodes):
            current, dist = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            
            if current in target_nodes:
                distances[current] = dist
                
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
                    
        return distances

class BatchDynamicGNNWithLabeling(nn.Module):
    """Dynamic GNN supporting batch processing (with Dropout + BatchNorm regularization)"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_labels, time_weighed=False, dropout=0.3):
        super().__init__()
        self.label_embedding = nn.Embedding(num_labels, 8)
        self.conv1 = GCNConv(in_channels + 8, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.time_weighed = time_weighed
        
    def forward(self, batch_data):
        label_emb = self.label_embedding(batch_data.node_labels)
        x_with_labels = torch.cat([batch_data.x, label_emb], dim=1)
        
        # GCN layer 1 + BatchNorm + ReLU + Dropout
        x = self.conv1(x_with_labels, batch_data.edge_index, 
                       edge_weight=batch_data.edge_attr.view(-1) if self.time_weighed else None)
        x = self.bn1(x).relu()
        x = self.dropout(x)
        
        # GCN layer 2 + BatchNorm (no ReLU, to stay consistent with the original output)
        x = self.conv2(x, batch_data.edge_index, 
                       edge_weight=batch_data.edge_attr.view(-1) if self.time_weighed else None)
        x = self.bn2(x)
        return x

class SEALWithTemporalWeighting(nn.Module):
    """SEAL framework for batch processing (with GNN Dropout + edge Dropout regularization)"""
    def __init__(self, num_features, hidden_dim, k_hop=2, batch_size=32, 
                 gcn_dropout=0.3, edge_dropout=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 message_direction='source_to_target',
                 use_fc_embedding=True, fc_embed_dim=128, fc_hidden_dim=None, fc_num_layers=3):
        super().__init__()
        # Direction switch, accepted only so that the shared Training/common_config.yaml
        # model.kwargs block can be handed to every backbone without special-casing SEAL.  SEAL
        # aggregates with GCNConv on k-hop subgraphs and GCNConv has no `flow` argument, so its
        # direction is fixed to source_to_target (the same side the new shared default uses).
        # Anything else would be silently ignored, so it is rejected loudly instead.
        if message_direction != 'source_to_target':
            raise ValueError(
                "Temp-SEAL aggregates through GCNConv on k-hop subgraphs, which has no `flow` "
                "argument and is therefore fixed to 'source_to_target'; got "
                f"message_direction={message_direction!r}."
            )
        self.batch_size = batch_size
        self.edge_dropout = edge_dropout
        self.subgraph_extractor = BatchSubgraphExtractor(k_hop=k_hop, batch_size=batch_size, device=device)

        # Shared node-feature extractor (same module/hyper-parameters in every backbone model).
        # It projects the subgraph node features before they meet the DRNL label embedding.
        self.feature_extractor = build_feature_extractor(
            input_dim=num_features,
            enabled=use_fc_embedding,
            output_dim=fc_embed_dim,
            hidden_dim=fc_hidden_dim,
            dropout=gcn_dropout,
            num_layers=fc_num_layers,
        )
        gnn_in_channels = fc_embed_dim if self.feature_extractor is not None else num_features

        self.dynamic_gnn = BatchDynamicGNNWithLabeling(
            gnn_in_channels, hidden_dim, hidden_dim, k_hop*2+2, 
            time_weighed=True, dropout=gcn_dropout
        ).to(device)
        self.static_gnn = BatchDynamicGNNWithLabeling(
            gnn_in_channels, hidden_dim, hidden_dim, k_hop*2+2,
            dropout=gcn_dropout
        ).to(device)
        self.device=device
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.to(self.device)
    
    def _drop_edges(self, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        """Randomly drop edges during training (GraphSAGE-style edge dropout), filtering edge_attr accordingly"""
        if self.edge_dropout <= 0 or not self.training:
            return edge_index, edge_attr
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) > self.edge_dropout
        edge_index = edge_index[:, keep_mask]
        if edge_attr is not None:
            edge_attr = edge_attr[keep_mask]
        return edge_index, edge_attr
        
    def forward(self, data, link_indices, current_times):
        """Batch-process multiple links: link_indices [B, 2]; current_times [B]. Returns the predicted probabilities."""
        batch_subgraphs = self.subgraph_extractor.batch_extract_subgraphs(
            data, link_indices, current_times
        )
        
        all_predictions = []
        
        for batch_data in batch_subgraphs:
            # Randomly drop subgraph edges during training (data augmentation, prevents overfitting)
            if self.training and self.edge_dropout > 0:
                edge_attr = batch_data.edge_attr if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None else None
                batch_data.edge_index, batch_data.edge_attr = self._drop_edges(batch_data.edge_index, edge_attr)
            
            # Project the subgraph node features once through the shared extractor,
            # so dynamic_gnn and static_gnn consume exactly the same representation.
            batch_data.x = apply_node_feature_extractor(self.feature_extractor, batch_data.x)

            # Batch GNN processing
            dynamic_embeddings = self.dynamic_gnn(batch_data)
            static_embeddings = self.static_gnn(batch_data)
            
            # Extract the embeddings of nodes A and B (use Batch.ptr to locate nodes within each subgraph)
            batch_size = batch_data.num_graphs
            ptr = batch_data.ptr  # [0, n0, n0+n1, ..., total_n]
            a_indices = ptr[:batch_size] + batch_data.a_mapping[:batch_size]
            b_indices = ptr[:batch_size] + batch_data.b_mapping[:batch_size]
            
            a_static = static_embeddings[a_indices]
            a_dynamic = dynamic_embeddings[a_indices]
            b_static = static_embeddings[b_indices]
            b_dynamic = dynamic_embeddings[b_indices]
            
            a_embeddings = torch.cat([a_static, a_dynamic], dim=1)
            b_embeddings = torch.cat([b_static, b_dynamic], dim=1)
            
            # Batch link prediction
            feature_vectors = torch.cat([
                a_embeddings, 
                b_embeddings, 
                current_times[:batch_size].unsqueeze(1)
            ], dim=1)
            
            batch_predictions = self.link_predictor(feature_vectors)
            all_predictions.append(batch_predictions.squeeze(-1))
        
        return torch.cat(all_predictions)
