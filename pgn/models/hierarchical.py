"""
Hierarchical Network for Comparison
"""

import torch
import torch.nn as nn
from typing import Optional, List


class HierarchicalNetwork(nn.Module):
    """
    Standard hierarchical network for comparison with PGN
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of layers
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 output_dim: int = 10, 
                 num_layers: int = 4,
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        layers = []
        dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Store individual layers for representation extraction
        self._build_layer_dict()
    
    def _build_layer_dict(self):
        """Build dictionary for accessing individual layers"""
        self.layer_dict = {}
        layer_idx = 0
        
        for i, module in enumerate(self.model):
            if isinstance(module, nn.Linear):
                self.layer_dict[f'layer_{layer_idx}'] = i
                layer_idx += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
    
    def get_representations(self, x: torch.Tensor) -> dict:
        """
        Get intermediate representations
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of representations at each layer
        """
        representations = {}
        
        for name, idx in self.layer_dict.items():
            # Forward up to this layer
            partial_model = self.model[:idx+1]
            rep = partial_model(x)
            representations[name] = rep
            
        representations['output'] = self.model(x)
        return representations
    
    def get_layer_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get outputs from all layers
        
        Args:
            x: Input tensor
            
        Returns:
            List of layer outputs
        """
        outputs = []
        current = x
        
        for module in self.model:
            current = module(current)
            if isinstance(module, nn.Linear):
                outputs.append(current.clone())
                
        return outputs