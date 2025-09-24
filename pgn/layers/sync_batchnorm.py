"""
Synchronized Batch Normalization Layer
"""

import torch
import torch.nn as nn
from typing import List


class SynchronizedBatchNorm(nn.Module):
    """
    Synchronized batch normalization across parallel branches
    Ensures temporal dynamics are shared while allowing geometric divergence
    
    Args:
        num_features: Number of features in input
        num_branches: Number of parallel branches to synchronize
        momentum: Momentum for running statistics
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, num_features: int, num_branches: int, 
                 momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_branches = num_branches
        self.momentum = momentum
        self.eps = eps
        
        # Shared temporal statistics
        self.register_buffer('shared_mean', torch.zeros(1, num_features))
        self.register_buffer('shared_var', torch.ones(1, num_features))
        
        # Branch-specific geometric transformations
        self.branch_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, num_features)) 
            for _ in range(num_branches)
        ])
        self.branch_shifts = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_features)) 
            for _ in range(num_branches)
        ])
        
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply synchronized normalization to all branches
        
        Args:
            inputs: List of tensors from each branch
            
        Returns:
            List of normalized tensors
        """
        # Compute shared statistics across all branches
        if self.training:
            all_inputs = torch.cat(inputs, dim=0)
            batch_mean = all_inputs.mean(dim=0, keepdim=True)
            batch_var = all_inputs.var(dim=0, keepdim=True)
            
            # Update running statistics with momentum
            self.shared_mean = (1 - self.momentum) * self.shared_mean + \
                               self.momentum * batch_mean
            self.shared_var = (1 - self.momentum) * self.shared_var + \
                              self.momentum * batch_var
        else:
            batch_mean = self.shared_mean
            batch_var = self.shared_var
        
        # Apply normalization with branch-specific transformations
        outputs = []
        for i, x in enumerate(inputs):
            # Shared temporal normalization
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            
            # Branch-specific geometric transformation
            x_transformed = x_norm * self.branch_scales[i] + self.branch_shifts[i]
            outputs.append(x_transformed)
            
        return outputs
    
    def extra_repr(self) -> str:
        return f'num_branches={self.num_branches}, momentum={self.momentum}, eps={self.eps}'