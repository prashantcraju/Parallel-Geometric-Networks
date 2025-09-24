"""
Unit tests for PGN models
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgn.models.pgn import ParallelGeometricNetwork
from pgn.models.hierarchical import HierarchicalNetwork
from pgn.models.geometric_branch import GeometricBranch
from pgn.layers.sync_batchnorm import SynchronizedBatchNorm
from pgn.utils.analysis import analyze_representations, compute_diversity_metrics


class TestGeometricBranch:
    """Test GeometricBranch model"""
    
    def test_initialization(self):
        """Test branch initialization with different geometric biases"""
        biases = ['sparse', 'orthogonal', 'gaussian', 'xavier', 'random']
        
        for bias in biases:
            branch = GeometricBranch(
                input_dim=100,
                hidden_dim=50,
                output_dim=10,
                geometric_bias=bias
            )
            assert branch is not None
            assert branch.geometric_bias == bias
    
    def test_forward_pass(self):
        """Test forward pass through branch"""
        branch = GeometricBranch(100, 50, 10)
        x = torch.randn(32, 100)
        output = branch(x)
        
        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()
    
    def test_representations(self):
        """Test getting intermediate representations"""
        branch = GeometricBranch(100, 50, 10)
        x = torch.randn(16, 100)
        reps = branch.get_representations(x)
        
        assert 'layer1' in reps
        assert 'layer2' in reps
        assert 'output' in reps
        assert reps['layer1'].shape == (16, 50)
        assert reps['layer2'].shape == (16, 50)
        assert reps['output'].shape == (16, 10)


class TestParallelGeometricNetwork:
    """Test main PGN model"""
    
    def test_initialization(self):
        """Test PGN initialization"""
        model = ParallelGeometricNetwork(
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
            num_branches=4
        )
        
        assert model.num_branches == 4
        assert len(model.branches) == 4
        assert model.use_synchronized_bn == True
    
    def test_forward_pass(self):
        """Test forward pass through PGN"""
        model = ParallelGeometricNetwork(784, 128, 10, 4)
        x = torch.randn(32, 784)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()
    
    def test_forward_with_branch_outputs(self):
        """Test getting branch outputs"""
        model = ParallelGeometricNetwork(784, 128, 10, 4)
        x = torch.randn(32, 784)
        output, branch_outputs = model(x, return_branch_outputs=True)
        
        assert output.shape == (32, 10)
        assert len(branch_outputs) == 4
        assert all(b.shape == (32, 128) for b in branch_outputs)
    
    def test_branch_representations(self):
        """Test getting branch representations"""
        model = ParallelGeometricNetwork(784, 128, 10, 4)
        x = torch.randn(16, 784)
        reps = model.get_branch_representations(x)
        
        assert len(reps) == 4
        assert all('layer1' in r for r in reps)
        assert all('layer2' in r for r in reps)
        assert all('output' in r for r in reps)
    
    def test_freeze_unfreeze_branches(self):
        """Test freezing and unfreezing branches"""
        model = ParallelGeometricNetwork(784, 128, 10, 4)
        
        # Count initial trainable params
        initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Freeze branches 0 and 1
        model.freeze_branches([0, 1])
        frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert frozen_trainable < initial_trainable
        
        # Unfreeze all
        model.unfreeze_branches()
        unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert unfrozen_trainable == initial_trainable
    
    def test_different_branch_counts(self):
        """Test PGN with different numbers of branches"""
        for num_branches in [1, 2, 4, 8]:
            model = ParallelGeometricNetwork(
                input_dim=100,
                hidden_dim=50,
                output_dim=10,
                num_branches=num_branches
            )
            
            x = torch.randn(16, 100)
            output = model(x)
            assert output.shape == (16, 10)
    
    def test_dropout(self):
        """Test PGN with dropout"""
        model = ParallelGeometricNetwork(
            input_dim=100,
            hidden_dim=50,
            output_dim=10,
            dropout_rate=0.5
        )
        
        # Set to training mode
        model.train()
        x = torch.randn(100, 100)
        
        # Get multiple outputs - should be different due to dropout
        output1 = model(x)
        output2 = model(x)
        
        assert not torch.allclose(output1, output2)
        
        # Set to eval mode - outputs should be same
        model.eval()
        output3 = model(x)
        output4 = model(x)
        
        assert torch.allclose(output3, output4)


class TestHierarchicalNetwork:
    """Test Hierarchical baseline model"""
    
    def test_initialization(self):
        """Test hierarchical network initialization"""
        model = HierarchicalNetwork(
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
            num_layers=4
        )
        
        assert model.num_layers == 4
        assert model.hidden_dim == 128
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = HierarchicalNetwork(784, 128, 10, 4)
        x = torch.randn(32, 784)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()
    
    def test_representations(self):
        """Test getting representations"""
        model = HierarchicalNetwork(784, 128, 10, 4)
        x = torch.randn(16, 784)
        reps = model.get_representations(x)
        
        assert 'output' in reps
        assert reps['output'].shape == (16, 10)
    
    def test_layer_outputs(self):
        """Test getting layer outputs"""
        model = HierarchicalNetwork(784, 128, 10, 4)
        x = torch.randn(16, 784)
        outputs = model.get_layer_outputs(x)
        
        assert len(outputs) == 4  # num_layers
        assert outputs[-1].shape == (16, 10)  # Final output


class TestSynchronizedBatchNorm:
    """Test Synchronized Batch Normalization"""
    
    def test_initialization(self):
        """Test SyncBN initialization"""
        sync_bn = SynchronizedBatchNorm(
            num_features=128,
            num_branches=4
        )
        
        assert sync_bn.num_branches == 4
        assert len(sync_bn.branch_scales) == 4
        assert len(sync_bn.branch_shifts) == 4
    
    def test_forward_pass(self):
        """Test forward pass through SyncBN"""
        sync_bn = SynchronizedBatchNorm(128, 4)
        
        # Create input for 4 branches
        inputs = [torch.randn(32, 128) for _ in range(4)]
        outputs = sync_bn(inputs)
        
        assert len(outputs) == 4
        assert all(o.shape == (32, 128) for o in outputs)
        assert all(not torch.isnan(o).any() for o in outputs)
    
    def test_training_vs_eval(self):
        """Test different behavior in training vs eval mode"""
        sync_bn = SynchronizedBatchNorm(128, 4)
        inputs = [torch.randn(32, 128) for _ in range(4)]
        
        # Training mode
        sync_bn.train()
        outputs_train = sync_bn(inputs)
        
        # Eval mode
        sync_bn.eval()
        outputs_eval = sync_bn(inputs)
        
        # Outputs should be different (using different statistics)
        for train, eval_out in zip(outputs_train, outputs_eval):
            assert not torch.allclose(train, eval_out, atol=1e-5)


class TestAnalysis:
    """Test analysis functions"""
    
    def test_analyze_representations(self):
        """Test representation analysis"""
        model = ParallelGeometricNetwork(100, 50, 10, 4)
        x = torch.randn(32, 100)
        
        analysis = analyze_representations(model, x)
        
        assert 'mean_geometric_distance' in analysis
        assert 'mean_temporal_correlation' in analysis
        assert 'branch_specialization' in analysis
        assert 'mean_effective_rank' in analysis
        
        assert 0 <= analysis['mean_geometric_distance'] <= 2
        assert analysis['mean_effective_rank'] > 0
    
    def test_diversity_metrics(self):
        """Test diversity metrics computation"""
        # Create diverse representations
        reps = [
            torch.randn(32, 128),
            torch.randn(32, 128) * 2,
            torch.randn(32, 128) * 0.5,
            torch.randn(32, 128) + 1
        ]
        
        metrics = compute_diversity_metrics(reps)
        
        assert 'mean_distance' in metrics
        assert 'distance_variance' in metrics
        assert 'mean_orthogonality' in metrics
        assert 'diversity_index' in metrics
        
        assert metrics['mean_distance'] > 0
        assert metrics['distance_variance'] >= 0


class TestModelComparison:
    """Test model comparison utilities"""
    
    def test_parameter_count(self):
        """Compare parameter counts between models"""
        pgn = ParallelGeometricNetwork(784, 128, 10, 4)
        hier = HierarchicalNetwork(784, 128, 10, 4)
        
        pgn_params = sum(p.numel() for p in pgn.parameters())
        hier_params = sum(p.numel() for p in hier.parameters())
        
        # PGN should have more parameters due to parallel branches
        assert pgn_params > hier_params
        
        print(f"PGN parameters: {pgn_params:,}")
        print(f"Hierarchical parameters: {hier_params:,}")
    
    def test_forward_speed(self):
        """Compare forward pass speed"""
        import time
        
        pgn = ParallelGeometricNetwork(784, 128, 10, 4)
        hier = HierarchicalNetwork(784, 128, 10, 4)
        
        x = torch.randn(100, 784)
        
        # Warm up
        _ = pgn(x)
        _ = hier(x)
        
        # Time PGN
        start = time.time()
        for _ in range(10):
            _ = pgn(x)
        pgn_time = time.time() - start
        
        # Time Hierarchical
        start = time.time()
        for _ in range(10):
            _ = hier(x)
        hier_time = time.time() - start
        
        print(f"PGN time: {pgn_time:.4f}s")
        print(f"Hierarchical time: {hier_time:.4f}s")
        
        # PGN might be slightly slower due to parallel processing
        assert pgn_time > 0 and hier_time > 0


@pytest.mark.parametrize("input_dim,hidden_dim,output_dim,num_branches", [
    (784, 128, 10, 4),
    (100, 50, 10, 2),
    (256, 64, 5, 8),
    (512, 256, 100, 3),
])
def test_pgn_configurations(input_dim, hidden_dim, output_dim, num_branches):
    """Test PGN with various configurations"""
    model = ParallelGeometricNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_branches=num_branches
    )
    
    batch_size = 16
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize("geometric_bias", ['sparse', 'orthogonal', 'gaussian', 'xavier'])
def test_geometric_biases(geometric_bias):
    """Test different geometric biases"""
    branch = GeometricBranch(
        input_dim=100,
        hidden_dim=50,
        output_dim=10,
        geometric_bias=geometric_bias
    )
    
    x = torch.randn(32, 100)
    output = branch(x)
    
    assert output.shape == (32, 10)
    assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])