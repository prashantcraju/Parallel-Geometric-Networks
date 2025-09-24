"""
Geometric Branch Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal


GeometricBiasType = Literal['sparse', 'orthogonal', 'gaussian', 'xavier', 'random']


class GeometricBranch(nn.Module):
    """
    Single branch with specific geometric bias
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        geometric_bias: Type of geometric initialization
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 geometric_bias: GeometricBiasType = 'random'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.geometric_bias = geometric_bias
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize with geometric bias
        self._initialize_geometry()
        
    def _initialize_geometry(self):
        """Apply geometric bias through initialization"""
        
        if self.geometric_bias == 'sparse':
            # Sparse initialization for local features
            for layer in [self.fc1, self.fc2, self.fc3]:
                nn.init.sparse_(layer.weight, sparsity=0.9)
                nn.init.zeros_(layer.bias)
                
        elif self.geometric_bias == 'orthogonal':
            # Orthogonal initialization for decorrelated features
            for layer in [self.fc1, self.fc2, self.fc3]:
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
        elif self.geometric_bias == 'gaussian':
            # Different variance for different scales
            nn.init.normal_(self.fc1.weight, std=0.1)
            nn.init.normal_(self.fc2.weight, std=0.5)
            nn.init.normal_(self.fc3.weight, std=1.0)
            for layer in [self.fc1, self.fc2, self.fc3]:
                nn.init.zeros_(layer.bias)
                
        elif self.geometric_bias == 'xavier':
            # Standard Xavier for balanced features
            for layer in [self.fc1, self.fc2, self.fc3]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        else:
            # Random/default initialization
            pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the branch
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_representations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate representations for analysis
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing representations at each layer
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        
        return {
            'layer1': h1,
            'layer2': h2,
            'output': h3,
            'pre_activation_1': self.fc1(x),
            'pre_activation_2': self.fc2(h1),
        }
    
    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'output_dim={self.output_dim}, geometric_bias={self.geometric_bias}')