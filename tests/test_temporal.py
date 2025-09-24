"""
Unit tests for Temporal PGN models and datasets
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgn.models.temporal_pgn import TemporalPGN, TemporalGeometricBranch
from pgn.models.hierarchical_rnn import HierarchicalRNN
from pgn.datasets.sequential_mnist import (
    SequentialMNIST, 
    PermutedMNIST, 
    NoisySequentialMNIST
)


class TestTemporalGeometricBranch:
    """Test Temporal Geometric Branch"""
    
    def test_initialization(self):
        """Test branch initialization"""
        branch = TemporalGeometricBranch(
            input_dim=28,
            hidden_dim=128,
            output_dim=128,
            geometric_bias='orthogonal',
            rnn_type='lstm'
        )
        
        assert branch.rnn_type == 'lstm'
        assert branch.geometric_bias == 'orthogonal'
        assert isinstance(branch.rnn, nn.LSTM)
    
    def test_forward_pass(self):
        """Test forward pass through temporal branch"""
        branch = TemporalGeometricBranch(28, 128, 128)
        
        # Input: (batch, time, features)
        x = torch.randn(32, 28, 28)
        output, temporal_output = branch(x)
        
        assert output.shape == (32, 128)  # Final output
        assert temporal_output.shape == (32, 28, 128)  # All timesteps
        assert not torch.isnan(output).any()
    
    def test_rnn_types(self):
        """Test different RNN types"""
        for rnn_type in ['lstm', 'gru', 'rnn']:
            branch = TemporalGeometricBranch(
                input_dim=28,
                hidden_dim=64,
                output_dim=64,
                rnn_type=rnn_type
            )
            
            x = torch.randn(16, 28, 28)
            output, _ = branch(x)
            assert output.shape == (16, 64)


class TestTemporalPGN:
    """Test Temporal PGN model"""
    
    def test_initialization(self):
        """Test TemporalPGN initialization"""
        model = TemporalPGN(
            input_dim=28,
            hidden_dim=128,
            output_dim=10,
            num_branches=4,
            rnn_type='lstm'
        )
        
        assert model.num_branches == 4
        assert len(model.branches) == 4
        assert isinstance(model.temporal_encoder, nn.LSTM)
    
    def test_forward_pass(self):
        """Test forward pass through TemporalPGN"""
        model = TemporalPGN(28, 128, 10, 4)
        
        # Sequential input
        x = torch.randn(32, 28, 28)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()
    
    def test_forward_with_dynamics(self):
        """Test getting temporal dynamics"""
        model = TemporalPGN(28, 128, 10, 4)
        x = torch.randn(16, 28, 28)
        
        output, temporal_features, branch_outputs = model(x, return_dynamics=True)
        
        assert output.shape == (16, 10)
        assert temporal_features.shape == (16, 28, 128)  # All timesteps
        assert len(branch_outputs) == 4
        assert all(b.shape == (16, 128) for b in branch_outputs)
    
    def test_synchronization_analysis(self):
        """Test synchronization analysis"""
        model = TemporalPGN(28, 128, 10, 4)
        
        # Create mock data loader
        x = torch.randn(64, 28, 28)
        y = torch.randint(0, 10, (64,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)
        
        metrics = model.analyze_synchronization(loader)
        
        assert 'temporal_synchronization' in metrics
        assert 'geometric_divergence' in metrics
        assert 'parallel_score' in metrics
        
        assert 0 <= metrics['temporal_synchronization'] <= 1
        assert 0 <= metrics['geometric_divergence'] <= 2
        assert metrics['parallel_score'] >= 0
    
    def test_branch_dynamics(self):
        """Test getting branch dynamics"""
        model = TemporalPGN(28, 128, 10, 4)
        x = torch.randn(8, 28, 28)
        
        dynamics = model.get_branch_dynamics(x)
        
        assert 'temporal_features' in dynamics
        assert 'branch_outputs' in dynamics
        assert 'branch_distances' in dynamics
        
        assert dynamics['temporal_features'].shape == (8, 28, 128)
        assert len(dynamics['branch_outputs']) == 4
        assert dynamics['branch_distances'].shape == (4, 4)
    
    def test_different_rnn_types(self):
        """Test TemporalPGN with different RNN types"""
        for rnn_type in ['lstm', 'gru', 'rnn']:
            model = TemporalPGN(
                input_dim=28,
                hidden_dim=64,
                output_dim=10,
                rnn_type=rnn_type
            )
            
            x = torch.randn(16, 28, 28)
            output = model(x)
            assert output.shape == (16, 10)


class TestHierarchicalRNN:
    """Test Hierarchical RNN baseline"""
    
    def test_initialization(self):
        """Test HierarchicalRNN initialization"""
        model = HierarchicalRNN(
            input_dim=28,
            hidden_dim=128,
            output_dim=10,
            num_layers=2,
            rnn_type='lstm'
        )
        
        assert model.num_layers == 2
        assert isinstance(model.rnn, nn.LSTM)
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = HierarchicalRNN(28, 128, 10, 2)
        x = torch.randn(32, 28, 28)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()
    
    def test_forward_with_hidden(self):
        """Test getting hidden states"""
        model = HierarchicalRNN(28, 128, 10, 2)
        x = torch.randn(16, 28, 28)
        
        output, rnn_out, hidden = model(x, return_hidden=True)
        
        assert output.shape == (16, 10)
        assert rnn_out.shape == (16, 28, 128)  # All timesteps
    
    def test_bidirectional(self):
        """Test bidirectional RNN"""
        model = HierarchicalRNN(
            input_dim=28,
            hidden_dim=64,
            output_dim=10,
            bidirectional=True
        )
        
        x = torch.randn(16, 28, 28)
        output = model(x)
        assert output.shape == (16, 10)
    
    def test_attention_weights(self):
        """Test attention weight computation"""
        model = HierarchicalRNN(28, 128, 10, 2)
        x = torch.randn(16, 28, 28)
        
        attention = model.compute_attention_weights(x)
        
        assert attention.shape == (16, 28)
        assert torch.allclose(attention.sum(dim=1), torch.ones(16), atol=1e-5)


class TestSequentialMNIST:
    """Test Sequential MNIST dataset"""
    
    @pytest.fixture
    def mnist_sample(self):
        """Create a mock MNIST sample"""
        image = torch.randn(1, 28, 28)
        label = 5
        return [(image, label)]
    
    def test_row_sequence(self, mnist_sample):
        """Test row-wise sequence generation"""
        dataset = SequentialMNIST(mnist_sample, sequence_type='row')
        seq, label = dataset[0]
        
        assert seq.shape == (28, 28)  # 28 timesteps, 28 features
        assert label == 5
    
    def test_column_sequence(self, mnist_sample):
        """Test column-wise sequence generation"""
        dataset = SequentialMNIST(mnist_sample, sequence_type='column')
        seq, label = dataset[0]
        
        assert seq.shape == (28, 28)
        assert label == 5
    
    def test_spiral_sequence(self, mnist_sample):
        """Test spiral sequence generation"""
        dataset = SequentialMNIST(mnist_sample, sequence_type='spiral')
        seq, label = dataset[0]
        
        assert seq.shape == (784, 1)  # 784 timesteps, 1 feature
        assert label == 5
    
    def test_random_sequence(self, mnist_sample):
        """Test random sequence generation"""
        dataset = SequentialMNIST(mnist_sample, sequence_type='random')
        seq, label = dataset[0]
        
        assert seq.shape == (784, 1)
        assert label == 5
    
    def test_with_noise(self, mnist_sample):
        """Test adding noise to sequences"""
        dataset = SequentialMNIST(
            mnist_sample, 
            sequence_type='row',
            noise_level=0.1
        )
        seq, label = dataset[0]
        
        assert seq.shape == (28, 28)
        assert label == 5


class TestPermutedMNIST:
    """Test Permuted MNIST dataset"""
    
    @pytest.fixture
    def mnist_sample(self):
        """Create a mock MNIST sample"""
        image = torch.randn(1, 28, 28)
        label = 3
        return [(image, label)]
    
    def test_fixed_permutation(self, mnist_sample):
        """Test fixed permutation across samples"""
        dataset = PermutedMNIST(
            mnist_sample,
            num_permutations=1,
            fixed_permutation=True
        )
        
        seq1, _ = dataset[0]
        assert seq1.shape == (28, 28)
        
        # Same permutation should be applied
        seq2, _ = dataset[0]
        assert torch.allclose(seq1, seq2)
    
    def test_multiple_permutations(self, mnist_sample):
        """Test multiple permutations"""
        dataset = PermutedMNIST(
            mnist_sample * 10,  # Replicate sample
            num_permutations=3,
            fixed_permutation=False
        )
        
        assert len(dataset) == 30  # 10 samples * 3 permutations


class TestNoisySequentialMNIST:
    """Test Noisy Sequential MNIST"""
    
    @pytest.fixture
    def mnist_sample(self):
        """Create a mock MNIST sample"""
        image = torch.randn(1, 28, 28)
        label = 7
        return [(image, label)]
    
    def test_noise_scheduling(self, mnist_sample):
        """Test progressive noise scheduling"""
        dataset = NoisySequentialMNIST(
            mnist_sample,
            noise_schedule='linear',
            max_noise=0.5
        )
        
        # Epoch 0 - minimal noise
        dataset.set_epoch(0)
        seq0, _ = dataset[0]
        
        # Epoch 10 - more noise
        dataset.set_epoch(10)
        seq10, _ = dataset[0]
        
        # Later epoch should have more noise (different values)
        assert not torch.allclose(seq0, seq10)
    
    def test_noise_schedules(self, mnist_sample):
        """Test different noise schedules"""
        for schedule in ['linear', 'exponential', 'step']:
            dataset = NoisySequentialMNIST(
                mnist_sample,
                noise_schedule=schedule,
                max_noise=0.3
            )
            
            dataset.set_epoch(5)
            seq, label = dataset[0]
            
            assert seq.shape == (28, 28)
            assert label == 7


class TestTemporalIntegration:
    """Integration tests for temporal models"""
    
    def test_temporal_pgn_vs_hierarchical_rnn(self):
        """Compare TemporalPGN and HierarchicalRNN"""
        # Create models
        pgn = TemporalPGN(28, 64, 10, 4)
        rnn = HierarchicalRNN(28, 64, 10, 2)
        
        # Same input
        x = torch.randn(16, 28, 28)
        
        # Forward pass
        pgn_output = pgn(x)
        rnn_output = rnn(x)
        
        assert pgn_output.shape == rnn_output.shape == (16, 10)
        
        # PGN should have more parameters
        pgn_params = sum(p.numel() for p in pgn.parameters())
        rnn_params = sum(p.numel() for p in rnn.parameters())
        assert pgn_params > rnn_params
    
    def test_training_step(self):
        """Test a single training step"""
        model = TemporalPGN(28, 64, 10, 4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Mock batch
        x = torch.randn(32, 28, 28)
        y = torch.randint(0, 10, (32,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_sequential_mnist_dataloader(self):
        """Test creating DataLoader from SequentialMNIST"""
        # Create mock MNIST data
        images = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        mock_mnist = list(zip(images, labels))
        
        # Create sequential dataset
        dataset = SequentialMNIST(mock_mnist, sequence_type='row')
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Get a batch
        batch_x, batch_y = next(iter(loader))
        
        assert batch_x.shape == (16, 28, 28)
        assert batch_y.shape == (16,)


@pytest.mark.parametrize("input_dim,hidden_dim,output_dim,num_branches", [
    (28, 64, 10, 4),
    (10, 32, 5, 2),
    (50, 128, 20, 8),
])
def test_temporal_pgn_configurations(input_dim, hidden_dim, output_dim, num_branches):
    """Test TemporalPGN with various configurations"""
    model = TemporalPGN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_branches=num_branches
    )
    
    batch_size = 16
    time_steps = 20
    x = torch.randn(batch_size, time_steps, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize("sequence_type", ['row', 'column', 'spiral', 'random'])
def test_sequence_types(sequence_type):
    """Test different sequence types"""
    # Mock MNIST sample
    image = torch.randn(1, 28, 28)
    label = 0
    mock_data = [(image, label)]
    
    dataset = SequentialMNIST(mock_data, sequence_type=sequence_type)
    seq, _ = dataset[0]
    
    if sequence_type in ['row', 'column']:
        assert seq.shape == (28, 28)
    else:
        assert seq.shape == (784, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])