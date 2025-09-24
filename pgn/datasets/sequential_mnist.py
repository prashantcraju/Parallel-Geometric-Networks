"""
Sequential MNIST Dataset for Temporal Tasks
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Literal, Optional


class SequentialMNIST(Dataset):
    """
    Convert MNIST to sequential format by reading pixels in sequence
    
    Args:
        mnist_dataset: Original MNIST dataset
        sequence_type: How to create sequences ('row', 'column', 'spiral', 'random')
        permutation: Optional fixed permutation for pixels
        noise_level: Amount of noise to add (0-1)
    """
    
    def __init__(self, 
                 mnist_dataset,
                 sequence_type: Literal['row', 'column', 'spiral', 'random'] = 'row',
                 permutation: Optional[np.ndarray] = None,
                 noise_level: float = 0.0):
        
        self.mnist_dataset = mnist_dataset
        self.sequence_type = sequence_type
        self.noise_level = noise_level
        
        # Create or use permutation
        if sequence_type == 'random' and permutation is None:
            self.permutation = np.random.permutation(784)
        elif permutation is not None:
            self.permutation = permutation
        else:
            self.permutation = self._create_sequence_order()
    
    def _create_sequence_order(self) -> np.ndarray:
        """Create sequence order based on type"""
        if self.sequence_type == 'row':
            # Read row by row
            return np.arange(784)
        
        elif self.sequence_type == 'column':
            # Read column by column
            indices = []
            for col in range(28):
                for row in range(28):
                    indices.append(row * 28 + col)
            return np.array(indices)
        
        elif self.sequence_type == 'spiral':
            # Read in spiral pattern
            return self._create_spiral_indices()
        
        else:  # random
            return np.random.permutation(784)
    
    def _create_spiral_indices(self) -> np.ndarray:
        """Create spiral reading pattern for 28x28 image"""
        indices = []
        matrix = np.arange(784).reshape(28, 28)
        
        top, bottom, left, right = 0, 28, 0, 28
        
        while top < bottom and left < right:
            # Read top row
            for i in range(left, right):
                indices.append(matrix[top][i])
            top += 1
            
            # Read right column
            for i in range(top, bottom):
                indices.append(matrix[i][right-1])
            right -= 1
            
            # Read bottom row
            if top < bottom:
                for i in range(right-1, left-1, -1):
                    indices.append(matrix[bottom-1][i])
                bottom -= 1
            
            # Read left column
            if left < right:
                for i in range(bottom-1, top-1, -1):
                    indices.append(matrix[i][left])
                left += 1
        
        return np.array(indices)
    
    def __len__(self) -> int:
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get sequential MNIST sample
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (sequence, label) where sequence is (28, 28) or (784, 1)
        """
        # Get original MNIST sample
        image, label = self.mnist_dataset[idx]
        
        # Flatten image
        flat_image = image.view(-1)
        
        # Apply permutation to create sequence
        sequence = flat_image[self.permutation]
        
        # Add noise if specified
        if self.noise_level > 0:
            noise = torch.randn_like(sequence) * self.noise_level
            sequence = sequence + noise
        
        # Reshape based on sequence type
        if self.sequence_type in ['row', 'column']:
            # Shape as (28, 28) - 28 timesteps, 28 features
            sequence = sequence.view(28, 28)
        else:
            # Shape as (784, 1) - 784 timesteps, 1 feature
            sequence = sequence.view(784, 1)
        
        return sequence, label


class PermutedMNIST(Dataset):
    """
    Permuted MNIST for testing invariance to input order
    
    Args:
        mnist_dataset: Original MNIST dataset
        num_permutations: Number of different permutations to use
        fixed_permutation: Use same permutation for all samples
    """
    
    def __init__(self,
                 mnist_dataset,
                 num_permutations: int = 1,
                 fixed_permutation: bool = True):
        
        self.mnist_dataset = mnist_dataset
        self.num_permutations = num_permutations
        self.fixed_permutation = fixed_permutation
        
        # Generate permutations
        self.permutations = [
            np.random.permutation(784) 
            for _ in range(num_permutations)
        ]
    
    def __len__(self) -> int:
        return len(self.mnist_dataset) * self.num_permutations
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get permuted MNIST sample
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (sequence, label)
        """
        # Determine which permutation and sample
        perm_idx = idx % self.num_permutations if not self.fixed_permutation else 0
        sample_idx = idx // self.num_permutations if not self.fixed_permutation else idx
        
        # Get original sample
        image, label = self.mnist_dataset[sample_idx]
        
        # Apply permutation
        flat_image = image.view(-1)
        permuted = flat_image[self.permutations[perm_idx]]
        
        # Reshape as sequence (28, 28)
        sequence = permuted.view(28, 28)
        
        return sequence, label


class NoisySequentialMNIST(SequentialMNIST):
    """
    Sequential MNIST with progressive noise for robustness testing
    
    Args:
        mnist_dataset: Original MNIST dataset
        noise_schedule: How noise increases ('linear', 'exponential', 'step')
        max_noise: Maximum noise level
    """
    
    def __init__(self,
                 mnist_dataset,
                 noise_schedule: Literal['linear', 'exponential', 'step'] = 'linear',
                 max_noise: float = 0.5,
                 **kwargs):
        
        super().__init__(mnist_dataset, **kwargs)
        self.noise_schedule = noise_schedule
        self.max_noise = max_noise
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update epoch for noise scheduling"""
        self.current_epoch = epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample with scheduled noise"""
        sequence, label = super().__getitem__(idx)
        
        # Calculate noise level based on schedule
        if self.noise_schedule == 'linear':
            noise_level = min(self.current_epoch / 20, 1.0) * self.max_noise
        elif self.noise_schedule == 'exponential':
            noise_level = (1 - np.exp(-self.current_epoch / 5)) * self.max_noise
        else:  # step
            noise_level = self.max_noise if self.current_epoch > 10 else 0
        
        # Add scheduled noise
        if noise_level > 0:
            noise = torch.randn_like(sequence) * noise_level
            sequence = sequence + noise
        
        return sequence, label