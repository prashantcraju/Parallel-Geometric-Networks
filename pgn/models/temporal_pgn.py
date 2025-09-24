"""
Temporal Parallel Geometric Network for Sequential Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from torch.utils.data import DataLoader

from ..layers.sync_batchnorm import SynchronizedBatchNorm
from .geometric_branch import GeometricBranch


class TemporalGeometricBranch(nn.Module):
    """
    Temporal branch with RNN processing and geometric bias
    
    Args:
        input_dim: Input dimension per timestep
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        geometric_bias: Type of geometric initialization
        rnn_type: Type of RNN ('lstm', 'gru', 'rnn')
    """
    
    def __init__(self, 
                 input_dim: int = 28,
                 hidden_dim: int = 128, 
                 output_dim: int = 128,
                 geometric_bias: str = 'random',
                 rnn_type: str = 'lstm'):
        super().__init__()
        
        self.rnn_type = rnn_type
        self.geometric_bias = geometric_bias
        
        # RNN layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        
        # Output projection with geometric bias
        self.projection = nn.Linear(hidden_dim, output_dim)
        
        # Apply geometric initialization
        self._initialize_geometry()
        
    def _initialize_geometry(self):
        """Apply geometric bias to weights"""
        if self.geometric_bias == 'sparse':
            nn.init.sparse_(self.projection.weight, sparsity=0.9)
        elif self.geometric_bias == 'orthogonal':
            nn.init.orthogonal_(self.projection.weight)
        elif self.geometric_bias == 'gaussian':
            nn.init.normal_(self.projection.weight, std=0.5)
        elif self.geometric_bias == 'xavier':
            nn.init.xavier_uniform_(self.projection.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, time, features)
            
        Returns:
            Tuple of (output, temporal_features)
        """
        # RNN processing
        temporal_output, _ = self.rnn(x)
        
        # Take last timestep and apply projection
        final_output = self.projection(temporal_output[:, -1, :])
        
        return final_output, temporal_output


class TemporalPGN(nn.Module):
    """
    Temporal PGN with synchronized dynamics and divergent geometry
    
    Args:
        input_dim: Input dimension per timestep
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_branches: Number of parallel branches
        geometric_biases: List of geometric biases for branches
        use_synchronized_bn: Whether to use synchronized batch normalization
        rnn_type: Type of RNN to use
    """
    
    def __init__(self, 
                 input_dim: int = 28,
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 num_branches: int = 4,
                 geometric_biases: Optional[List[str]] = None,
                 use_synchronized_bn: bool = True,
                 rnn_type: str = 'lstm'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.use_synchronized_bn = use_synchronized_bn
        
        if geometric_biases is None:
            geometric_biases = ['sparse', 'orthogonal', 'gaussian', 'xavier']
        
        # Shared temporal encoder (synchronizes temporal dynamics)
        if rnn_type == 'lstm':
            self.temporal_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'gru':
            self.temporal_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            self.temporal_encoder = nn.RNN(input_dim, hidden_dim, batch_first=True)
        
        # Parallel branches with different geometric biases
        self.branches = nn.ModuleList([
            TemporalGeometricBranch(
                hidden_dim, hidden_dim, hidden_dim,
                geometric_biases[i % len(geometric_biases)],
                rnn_type='linear'  # Use linear for branch processing
            )
            for i in range(num_branches)
        ])
        
        # Branch-specific projections (for geometric divergence)
        self.branch_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_branches)
        ])
        
        # Synchronized batch normalization
        if use_synchronized_bn:
            self.sync_bn = SynchronizedBatchNorm(hidden_dim, num_branches)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * num_branches, output_dim)
        
        # Initialize with different geometric biases
        self._initialize_branches()
        
    def _initialize_branches(self):
        """Initialize branch projections with geometric biases"""
        init_funcs = [
            lambda w: nn.init.sparse_(w, sparsity=0.9),
            lambda w: nn.init.orthogonal_(w),
            lambda w: nn.init.normal_(w, std=0.5),
            lambda w: nn.init.xavier_uniform_(w)
        ]
        
        for i, projection in enumerate(self.branch_projections):
            for layer in projection:
                if isinstance(layer, nn.Linear):
                    init_funcs[i % len(init_funcs)](layer.weight)
    
    def forward(self, 
                x: torch.Tensor, 
                return_dynamics: bool = False
                ) -> Any:
        """
        Forward pass through temporal PGN
        
        Args:
            x: Input tensor of shape (batch, time, features)
            return_dynamics: Whether to return temporal dynamics and branch outputs
            
        Returns:
            Output tensor, optionally with dynamics
        """
        # Shared temporal encoding (synchronized dynamics)
        temporal_features, _ = self.temporal_encoder(x)
        
        # Process through parallel branches (divergent geometry)
        branch_outputs = []
        for i, projection in enumerate(self.branch_projections):
            # Apply branch-specific geometric transformation
            branch_out = projection(temporal_features[:, -1, :])
            branch_outputs.append(branch_out)
        
        # Apply synchronized normalization
        if self.use_synchronized_bn:
            branch_outputs = self.sync_bn(branch_outputs)
        
        # Concatenate and fuse
        combined = torch.cat(branch_outputs, dim=-1)
        output = self.fusion(combined)
        
        if return_dynamics:
            return output, temporal_features, branch_outputs
        return output
    
    def analyze_synchronization(self, 
                                data_loader: DataLoader,
                                device: str = 'cpu') -> Dict[str, float]:
        """
        Analyze temporal synchronization and geometric divergence
        
        Args:
            data_loader: DataLoader for analysis
            device: Device to use
            
        Returns:
            Dictionary of synchronization metrics
        """
        self.eval()
        
        all_temporal_corrs = []
        all_geometric_dists = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                
                # Get temporal and branch features
                _, temporal_features, branch_outputs = self.forward(
                    data[:32], return_dynamics=True  # Use subset for efficiency
                )
                
                # Compute temporal synchronization (correlation across time)
                temp_flat = temporal_features.reshape(-1, temporal_features.shape[-1])
                temporal_corr = torch.corrcoef(temp_flat.T).abs().mean().item()
                all_temporal_corrs.append(temporal_corr)
                
                # Compute geometric divergence (distance between branches)
                geometric_dists = []
                for i in range(self.num_branches):
                    for j in range(i+1, self.num_branches):
                        dist = F.cosine_similarity(
                            branch_outputs[i], 
                            branch_outputs[j], 
                            dim=1
                        ).mean().item()
                        geometric_dists.append(1 - dist)  # Convert similarity to distance
                
                all_geometric_dists.append(np.mean(geometric_dists))
                
                # Only analyze first batch for efficiency
                break
        
        temporal_sync = np.mean(all_temporal_corrs)
        geometric_div = np.mean(all_geometric_dists)
        
        # Parallel processing score (high sync + high divergence)
        parallel_score = temporal_sync * geometric_div
        
        return {
            'temporal_synchronization': temporal_sync,
            'geometric_divergence': geometric_div,
            'parallel_score': parallel_score
        }
    
    def get_branch_dynamics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed dynamics from each branch
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of branch dynamics
        """
        _, temporal_features, branch_outputs = self.forward(x, return_dynamics=True)
        
        return {
            'temporal_features': temporal_features,
            'branch_outputs': branch_outputs,
            'branch_distances': self._compute_branch_distances(branch_outputs)
        }
    
    def _compute_branch_distances(self, 
                                  branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute pairwise distances between branch outputs"""
        num_branches = len(branch_outputs)
        distances = torch.zeros(num_branches, num_branches)
        
        for i in range(num_branches):
            for j in range(num_branches):
                if i != j:
                    distances[i, j] = F.pairwise_distance(
                        branch_outputs[i], 
                        branch_outputs[j]
                    ).mean()
        
        return distances