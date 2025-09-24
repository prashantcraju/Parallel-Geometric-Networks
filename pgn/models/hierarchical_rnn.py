"""
Hierarchical RNN for comparison with Temporal PGN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any


class HierarchicalRNN(nn.Module):
    """
    Standard hierarchical RNN architecture for baseline comparison
    
    Args:
        input_dim: Input dimension per timestep
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of RNN layers
        rnn_type: Type of RNN ('lstm', 'gru', 'rnn')
        dropout_rate: Dropout rate between layers
        bidirectional: Whether to use bidirectional RNN
    """
    
    def __init__(self, 
                 input_dim: int = 28,
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 num_layers: int = 2,
                 rnn_type: str = 'lstm',
                 dropout_rate: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # RNN layers
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.RNN(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layers
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(rnn_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, 
                x: torch.Tensor,
                return_hidden: bool = False) -> Any:
        """
        Forward pass through hierarchical RNN
        
        Args:
            x: Input tensor of shape (batch, time, features)
            return_hidden: Whether to return hidden states
            
        Returns:
            Output tensor, optionally with hidden states
        """
        # RNN forward pass
        rnn_out, hidden = self.rnn(x)
        
        # Take the last timestep output
        if self.bidirectional:
            # Concatenate forward and backward final states
            last_output = rnn_out[:, -1, :]
        else:
            last_output = rnn_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(last_output))
        x = self.batch_norm(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        if return_hidden:
            return output, rnn_out, hidden
        return output
    
    def get_representations(self, x: torch.Tensor) -> dict:
        """
        Get intermediate representations for analysis
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of representations at different layers
        """
        rnn_out, hidden = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        
        fc1_out = F.relu(self.fc1(last_output))
        fc1_bn = self.batch_norm(fc1_out)
        output = self.fc2(fc1_bn)
        
        return {
            'rnn_output': rnn_out,
            'last_hidden': last_output,
            'fc1_output': fc1_out,
            'fc1_bn': fc1_bn,
            'final_output': output
        }
    
    def compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-like weights over timesteps
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights
        """
        rnn_out, _ = self.rnn(x)
        
        # Simple attention mechanism
        # Score each timestep
        scores = torch.mean(rnn_out, dim=2)  # (batch, time)
        attention_weights = F.softmax(scores, dim=1)
        
        return attention_weights