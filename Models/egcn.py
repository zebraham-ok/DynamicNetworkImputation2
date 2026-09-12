import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import math

from Models.common import build_feature_extractor, apply_node_feature_extractor


class EGCN_PyG(nn.Module):
    def __init__(self, dynamic_data, hidden_dims, activation=nn.ReLU(), 
                 num_gru_layers=2, dropout=0.3, time_steps=None, device=None):
        """hidden_dims: list of hidden dims per layer, e.g. [input_dim, hidden1, hidden2]."""
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_steps = time_steps if time_steps else list(range(dynamic_data.edge_time.min(), 
                                                                   dynamic_data.edge_time.max()+1))
        self.year_to_idx = {year: idx for idx, year in enumerate(self.time_steps)}
        
        self.num_nodes = dynamic_data.num_nodes
        
        # Build temporal subgraphs
        self.subgraphs = self._build_temporal_subgraphs(dynamic_data)
        
        # Initialize GRCU layers
        self.grcu_layers = nn.ModuleList()
        self.hidden_dims = hidden_dims
        
        for i in range(1, len(hidden_dims)):
            grcu_layer = GRCU_PyG(
                in_feats=hidden_dims[i-1],
                out_feats=hidden_dims[i],
                activation=activation,
                dropout=dropout
            )
            self.grcu_layers.append(grcu_layer)
        
        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dims[-1] + 1, 256),  # two node embeddings + time feature
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
    
    def _build_temporal_subgraphs(self, dynamic_data):
        """Build the list of temporal subgraphs"""
        subgraphs = []
        for t in self.time_steps:
            # Filter edges belonging to the current time step
            mask = dynamic_data.edge_time == t
            filtered_edge_index = dynamic_data.edge_index[:, mask]
            
            # Create subgraph data
            subgraph_kwargs = {
                'x': dynamic_data.x,
                'edge_index': filtered_edge_index,
                'num_nodes': self.num_nodes
            }
            
            # Add only when edge_attr exists and is not None
            if hasattr(dynamic_data, 'edge_attr') and dynamic_data.edge_attr is not None:
                subgraph_kwargs['edge_attr'] = dynamic_data.edge_attr[mask]
            
            subgraph_data = Data(**subgraph_kwargs).to(self.device)
            subgraphs.append(subgraph_data)
        
        return subgraphs
    
    def forward(self, node_pairs, time_indices):
        """node_pairs: [B, 2] node pairs; time_indices: [B] time-step years. Returns the probability that each edge exists [B]."""
        # Initialize the node feature sequence
        node_feats_seq = []
        for subgraph in self.subgraphs:
            node_feats_seq.append(subgraph.x.unsqueeze(0))  # [1, num_nodes, feat_dim]
        
        # Stack into a sequence [num_time_steps, num_nodes, feat_dim]
        node_feats_seq = torch.cat(node_feats_seq, dim=0)
        
        # Pass through the GRCU layers
        for grcu_layer in self.grcu_layers:
            node_feats_seq = grcu_layer(self.subgraphs, node_feats_seq)
        
        batch_size = len(time_indices)
        
        if torch.is_tensor(time_indices):
            time_years = time_indices.cpu().numpy()
        else:
            time_years = time_indices
        
        # Convert to indices (for years not in the list, take the closest one)
        time_positions = []
        for year in time_years:
            if year in self.year_to_idx:
                time_positions.append(self.year_to_idx[year])
            else:
                # Find the closest year
                closest_idx = min(range(len(self.time_steps)), 
                                key=lambda i: abs(self.time_steps[i] - year))
                time_positions.append(closest_idx)
        
        time_positions = torch.tensor(time_positions, device=self.device, dtype=torch.long)
        
        # Ensure node_pairs is a tensor
        if not torch.is_tensor(node_pairs):
            node_pairs = torch.tensor(node_pairs, device=self.device, dtype=torch.long)
        
        u_indices = node_pairs[:, 0].long()
        v_indices = node_pairs[:, 1].long()
        
        # Batch-fetch node embeddings
        emb_u = node_feats_seq[time_positions, u_indices]  # [batch_size, hidden_dim]
        emb_v = node_feats_seq[time_positions, v_indices]  # [batch_size, hidden_dim]
        
        time_normalized = time_positions.float() / len(self.time_steps)
        time_normalized = time_normalized.unsqueeze(1)  # [batch_size, 1]
        
        pair_features = torch.cat([emb_u, emb_v, time_normalized], dim=1)
        predictions = self.edge_predictor(pair_features).squeeze(1)
        
        return predictions

class EGCN_PyG_Vectorized(nn.Module):
    """Vectorized version of EGCN, precomputes all adjacency matrices, plus the shared node-feature extractor"""
    def __init__(self, dynamic_data, hidden_dims, activation=nn.ReLU(), 
                 num_gru_layers=2, dropout=0.3, time_steps=None, device=None,
                 use_fc_embedding=True, fc_embed_dim=128, fc_hidden_dim=None, fc_num_layers=3):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_steps = time_steps if time_steps else list(range(dynamic_data.edge_time.min(), 
                                                                   dynamic_data.edge_time.max()+1))
        self.year_to_idx = {year: idx for idx, year in enumerate(self.time_steps)}
        
        self.num_nodes = dynamic_data.num_nodes
        node_feat_dim = dynamic_data.x.size(1)
        
        # Shared node-feature extractor (same module/hyper-parameters in every backbone model)
        self.use_fc_embedding = use_fc_embedding
        self.feature_extractor = build_feature_extractor(
            input_dim=node_feat_dim,
            enabled=use_fc_embedding,
            output_dim=fc_embed_dim,
            hidden_dim=fc_hidden_dim,
            dropout=dropout,
            num_layers=fc_num_layers,
        )
        self.raw_node_feats = dynamic_data.x.to(self.device)
        if self.feature_extractor is not None:
            hidden_dims = [fc_embed_dim] + list(hidden_dims[1:])
        
        # Precompute the sparse adjacency matrices for all time steps
        self.adj_matrices = self._precompute_adj_matrices(dynamic_data)
        
        # Initialize GRCU layers
        self.grcu_layers = nn.ModuleList()
        self.hidden_dims = hidden_dims
        
        for i in range(1, len(hidden_dims)):
            grcu_layer = GRCU_PyG_Vectorized(
                in_feats=hidden_dims[i-1],
                out_feats=hidden_dims[i],
                activation=activation,
                dropout=dropout
            )
            self.grcu_layers.append(grcu_layer)
        
        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dims[-1] + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
    
    def _precompute_adj_matrices(self, dynamic_data):
        """Precompute the sparse adjacency matrices for all time steps"""
        adj_matrices = []
        
        for t in self.time_steps:
            mask = dynamic_data.edge_time == t
            filtered_edge_index = dynamic_data.edge_index[:, mask]
            
            if filtered_edge_index.size(1) > 0:
                # Symmetric normalization coefficients D^{-1/2} A D^{-1/2}
                row, col = filtered_edge_index
                num_edges = row.size(0)
                
                deg = torch.bincount(row, minlength=self.num_nodes).float()
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                
                adj = torch.sparse_coo_tensor(
                    filtered_edge_index,
                    norm,
                    size=(self.num_nodes, self.num_nodes)
                )
            else:
                # Adjacency matrix for an empty time step
                adj = torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long),
                    torch.empty((0,)),
                    size=(self.num_nodes, self.num_nodes)
                )
            
            adj_matrices.append(adj.coalesce().to(self.device))
        
        return adj_matrices
    
    def forward(self, node_pairs, time_indices):
        # Recompute node_feats on every forward to avoid backward-graph reuse errors
        node_feats = apply_node_feature_extractor(self.feature_extractor, self.raw_node_feats)
        
        # All time steps share the same node features
        node_feats_seq = node_feats.unsqueeze(0).expand(len(self.time_steps), -1, -1)
        
        for grcu_layer in self.grcu_layers:
            node_feats_seq = grcu_layer(self.adj_matrices, node_feats_seq)
        
        # Convert years to time-step indices
        if torch.is_tensor(time_indices):
            time_years = time_indices.cpu().numpy()
        else:
            time_years = time_indices
        
        time_positions = []
        for year in time_years:
            if year in self.year_to_idx:
                time_positions.append(self.year_to_idx[year])
            else:
                closest_idx = min(range(len(self.time_steps)), 
                                key=lambda i: abs(self.time_steps[i] - year))
                time_positions.append(closest_idx)
        
        time_positions = torch.tensor(time_positions, device=self.device, dtype=torch.long)
        
        if not torch.is_tensor(node_pairs):
            node_pairs = torch.tensor(node_pairs, device=self.device, dtype=torch.long)
        
        u_indices = node_pairs[:, 0].long()
        v_indices = node_pairs[:, 1].long()
        
        # Batch-fetch node embeddings
        emb_u = node_feats_seq[time_positions, u_indices]
        emb_v = node_feats_seq[time_positions, v_indices]
        
        time_normalized = time_positions.float() / len(self.time_steps)
        time_normalized = time_normalized.unsqueeze(1)
        
        pair_features = torch.cat([emb_u, emb_v, time_normalized], dim=1)
        predictions = self.edge_predictor(pair_features).squeeze(1)
        
        return predictions

class GRCU_PyG_Vectorized(nn.Module):
    """Vectorized version of the GRCU layer, using precomputed sparse adjacency matrices"""
    def __init__(self, in_feats, out_feats, activation, dropout=0.3):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        
        self.evolve_weights = MatGRUCell_PyG(rows=in_feats, cols=out_feats)
        
        self.gcn_init_weights = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.reset_param(self.gcn_init_weights)
        
        self.dropout = nn.Dropout(dropout)
    
    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)
    
    def forward(self, adj_matrices, node_feats_seq):
        num_timesteps = len(adj_matrices)
        gcn_weights = self.gcn_init_weights
        out_seq = []
        
        for t in range(num_timesteps):
            node_feats = node_feats_seq[t]
            adj_matrix = adj_matrices[t]
            
            gcn_weights = self.evolve_weights(gcn_weights, node_feats)
            transformed = node_feats @ gcn_weights
            
            if adj_matrix._nnz() > 0:
                aggregated = torch.sparse.mm(adj_matrix, transformed)
                out = aggregated + transformed  # self-connection
            else:
                out = transformed
            
            out = self.activation(out)
            out = self.dropout(out)
            
            out_seq.append(out.unsqueeze(0))
        
        return torch.cat(out_seq, dim=0)

class GRCU_PyG(nn.Module):
    """PyG version of the GRCU layer"""
    def __init__(self, in_feats, out_feats, activation, dropout=0.3):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = activation
        
        self.evolve_weights = MatGRUCell_PyG(rows=in_feats, cols=out_feats)
        
        self.gcn_init_weights = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.reset_param(self.gcn_init_weights)
        
        self.dropout = nn.Dropout(dropout)
    
    def reset_param(self, t):
        """Parameter initialization"""
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)
    
    def forward(self, subgraphs, node_feats_seq):
        """subgraphs: list of temporal subgraphs; node_feats_seq: [num_timesteps, num_nodes, in_feats].
        Returns the updated feature sequence [num_timesteps, num_nodes, out_feats]."""
        num_timesteps = len(subgraphs)
        num_nodes = node_feats_seq.size(1)
        
        gcn_weights = self.gcn_init_weights
        out_seq = []
        
        for t in range(num_timesteps):
            node_feats = node_feats_seq[t]  # [num_nodes, in_feats]
            
            gcn_weights = self.evolve_weights(gcn_weights, node_feats)
            
            subgraph = subgraphs[t]
            edge_index = subgraph.edge_index
            num_edges = edge_index.size(1)
            
            if num_edges > 0:
                # GCN symmetric normalization coefficients
                row, col = edge_index
                deg = torch.zeros(num_nodes, device=node_feats.device)
                deg = deg.scatter_add_(0, row, torch.ones(num_edges, device=node_feats.device))
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                
                transformed = node_feats @ gcn_weights  # [num_nodes, out_feats]
                
                # Manually implement GCN message passing
                out = torch.zeros_like(transformed)
                for i in range(num_edges):
                    src, dst = col[i], row[i]
                    out[dst] += transformed[src] * norm[i]
                
                out = out + transformed  # self-connection
                
                out = self.activation(out)
                out = self.dropout(out)
            else:
                # Without edges, apply the linear transform only
                out = node_feats @ gcn_weights
                out = self.activation(out)
                out = self.dropout(out)
            
            out_seq.append(out.unsqueeze(0))
        
        return torch.cat(out_seq, dim=0)


class MatGRUCell_PyG(nn.Module):
    """PyG version of the matrix GRU cell"""
    def __init__(self, rows, cols):
        super().__init__()
        self.rows = rows
        self.cols = cols
        
        # Three gates
        self.update_gate = MatGRUGate_PyG(rows, cols, torch.nn.Sigmoid())
        self.reset_gate = MatGRUGate_PyG(rows, cols, torch.nn.Sigmoid())
        self.candidate_gate = MatGRUGate_PyG(rows, cols, torch.nn.Tanh())
        
        self.use_topk = False
        if self.use_topk:
            self.topk = TopK_PyG(feats=rows, k=cols)
    
    def forward(self, prev_weights, node_feats, mask=None):
        """prev_weights: [rows, cols] weights of the previous time step; node_feats: [num_nodes, rows]. Returns the new weights [rows, cols]."""
        # Aggregate node features (take the mean)
        aggregated = torch.mean(node_feats, dim=0, keepdim=True)  # [1, rows]
        
        if self.use_topk and mask is not None:
            z_topk = self.topk(aggregated, mask)
        else:
            z_topk = aggregated.t()  # [rows, 1]
        
        update = self.update_gate(z_topk, prev_weights)
        reset = self.reset_gate(z_topk, prev_weights)
        
        h_cap = reset * prev_weights
        h_cap = self.candidate_gate(z_topk, h_cap)
        
        new_weights = (1 - update) * prev_weights + update * h_cap
        
        return new_weights


class MatGRUGate_PyG(nn.Module):
    """PyG version of the matrix GRU gate"""
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        
        self.W = nn.Parameter(torch.Tensor(rows, rows))
        self.U = nn.Parameter(torch.Tensor(rows, rows))
        self.bias = nn.Parameter(torch.zeros(rows, cols))
        
        self.reset_param(self.W)
        self.reset_param(self.U)
    
    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)
    
    def forward(self, x, hidden):
        # x: [rows, 1] or [rows, cols]; hidden: [rows, cols]
        if x.dim() == 2 and x.size(1) == 1:
            x = x.expand(-1, hidden.size(1))
        
        out = self.activation(
            self.W @ x + 
            self.U @ hidden + 
            self.bias
        )
        return out


class TopK_PyG(nn.Module):
    """PyG version of TopK selection"""
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = nn.Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)
        self.k = k
    
    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)
    
    def forward(self, node_embs, mask):
        # node_embs: [1, feats]
        scores = node_embs @ self.scorer / self.scorer.norm()
        
        if mask is not None:
            scores = scores + mask
        
        vals, indices = torch.topk(scores.view(-1), min(self.k, scores.numel()))
        
        out = torch.zeros(self.k, node_embs.size(1), device=node_embs.device)
        num_selected = indices.size(0)
        out[:num_selected] = node_embs[0, indices]
        
        return out.t()  # [feats, k]