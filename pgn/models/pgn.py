"""
Parallel Geometric Network (PGN) Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from ..layers.sync_batchnorm import SynchronizedBatchNorm
from .geometric_branch import GeometricBranch, GeometricBiasType


class ParallelGeometricNetwork(nn.Module):
    """
    Main PGN architecture with synchronized temporal dynamics
    and divergent geometric representations
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_branches: Number of parallel branches
        geometric_biases: List of geometric bias types for each branch
        use_synchronized_bn: Whether to use synchronized batch normalization
        use_temporal: Whether to use temporal processing (for sequential data)
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 num_branches: int = 4,
                 geometric_biases: Optional[List[GeometricBiasType]] = None,
                 use_synchronized_bn: bool = True,
                 use_temporal: bool = False,
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.use_synchronized_bn = use_synchronized_bn
        self.use_temporal = use_temporal
        
        # Default geometric biases
        if geometric_biases is None:
            geometric_biases = ['sparse', 'orthogonal', 'gaussian', 'xavier']
        
        # Create parallel branches with different geometric biases
        self.branches = nn.ModuleList([
            GeometricBranch(
                input_dim, 
                hidden_dim, 
                hidden_dim,
                geometric_biases[i % len(geometric_biases)]
            )
            for i in range(num_branches)
        ])
        
        # Synchronized batch normalization
        if use_synchronized_bn:
            self.sync_bn = SynchronizedBatchNorm(hidden_dim, num_branches)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * num_branches, output_dim)
        
        # Optional temporal processor for sequential data
        if use_temporal:
            self.temporal_processor = nn.GRU(
                hidden_dim, 
                hidden_dim, 
                batch_first=True,
                num_layers=1
            )
        
    def forward(self, 
                x: torch.Tensor, 
                return_branch_outputs: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through parallel branches
        
        Args:
            x: Input tensor
            return_branch_outputs: Whether to return individual branch outputs
            
        Returns:
            Output tensor, optionally with branch outputs
        """
        # Process through each branch
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Apply synchronized normalization
        if self.use_synchronized_bn:
            branch_outputs = self.sync_bn(branch_outputs)
        
        # Optional: Share temporal dynamics for sequential data
        if self.use_temporal and len(x.shape) == 3:  # (batch, time, features)
            processed_outputs = []
            for branch_out in branch_outputs:
                temp_out, _ = self.temporal_processor(branch_out)
                # Take last timestep
                processed_outputs.append(temp_out[:, -1, :])
            branch_outputs = processed_outputs
        
        # Apply dropout
        branch_outputs = [self.dropout(out) for out in branch_outputs]
        
        # Concatenate and fuse
        combined = torch.cat(branch_outputs, dim=-1)
        output = self.fusion(combined)
        
        if return_branch_outputs:
            return output, branch_outputs
        return output
    
    def get_branch_representations(self, x: torch.Tensor) -> List[dict]:
        """
        Get representations from all branches
        
        Args:
            x: Input tensor
            
        Returns:
            List of representation dictionaries from each branch
        """
        representations = []
        for branch in self.branches:
            reps = branch.get_representations(x)
            representations.append(reps)
        return representations
    
    def freeze_branches(self, branch_indices: Optional[List[int]] = None):
        """
        Freeze specific branches for fine-tuning
        
        Args:
            branch_indices: Indices of branches to freeze. If None, freeze all.
        """
        if branch_indices is None:
            branch_indices = list(range(self.num_branches))
            
        for idx in branch_indices:
            for param in self.branches[idx].parameters():
                param.requires_grad = False
    
    def unfreeze_branches(self, branch_indices: Optional[List[int]] = None):
        """
        Unfreeze specific branches
        
        Args:
            branch_indices: Indices of branches to unfreeze. If None, unfreeze all.
        """
        if branch_indices is None:
            branch_indices = list(range(self.num_branches))
            
        for idx in branch_indices:
            for param in self.branches[idx].parameters():
                param.requires_grad = True