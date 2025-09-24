"""
PGN Tutorial: Complete Examples
This file can be run as a Python script or converted to Jupyter notebook
"""

# %% [markdown]
# # Parallel Geometric Networks Tutorial
# 
# This tutorial demonstrates:
# 1. Basic PGN usage
# 2. Temporal PGN for sequential data
# 3. Analysis and visualization
# 4. Comparison with baselines

# %% Import libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import PGN modules
from pgn import (
    ParallelGeometricNetwork,
    TemporalPGN,
    HierarchicalNetwork,
    HierarchicalRNN,
    SequentialMNIST,
    Trainer
)
from pgn.utils.analysis import analyze_representations
from pgn.utils.visualization import visualize_branch_representations

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## 1. Basic PGN Example

# %% Basic PGN setup
def example_basic_pgn():
    """Example 1: Basic PGN for image classification"""
    print("="*60)
    print("Example 1: Basic PGN")
    print("="*60)
    
    # Create model
    model = ParallelGeometricNetwork(
        input_dim=784,  # MNIST flattened
        hidden_dim=128,
        output_dim=10,
        num_branches=4,
        geometric_biases=['sparse', 'orthogonal', 'gaussian', 'xavier'],
        use_synchronized_bn=True
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example forward pass
    batch_size = 32
    x = torch.randn(batch_size, 784)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get branch representations
    branch_reps = model.get_branch_representations(x)
    print(f"Number of branches: {len(branch_reps)}")
    print(f"Branch output shape: {branch_reps[0]['output'].shape}")
    
    # Analyze representations
    analysis = analyze_representations(model, x)
    print("\nRepresentation Analysis:")
    print(f"  Mean Geometric Distance: {analysis['mean_geometric_distance']:.4f}")
    print(f"  Branch Specialization: {analysis['branch_specialization']:.4f}")
    print(f"  Mean Effective Rank: {analysis['mean_effective_rank']:.2f}")
    
    return model

# %% [markdown]
# ## 2. Temporal PGN Example

# %% Temporal PGN setup
def example_temporal_pgn():
    """Example 2: Temporal PGN for sequential data"""
    print("\n" + "="*60)
    print("Example 2: Temporal PGN")
    print("="*60)
    
    # Create temporal model
    model = TemporalPGN(
        input_dim=28,  # 28 features per timestep
        hidden_dim=128,
        output_dim=10,
        num_branches=4,
        rnn_type='lstm'
    )
    
    print(f"Temporal model created")
    
    # Example sequential input (batch_size, time_steps, features)
    batch_size = 16
    time_steps = 28
    features = 28
    
    x = torch.randn(batch_size, time_steps, features)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get temporal dynamics
    output, temporal_features, branch_outputs = model(x, return_dynamics=True)
    
    print(f"Temporal features shape: {temporal_features.shape}")
    print(f"Number of branch outputs: {len(branch_outputs)}")
    
    # Analyze synchronization (mock data loader)
    from torch.utils.data import TensorDataset
    mock_dataset = TensorDataset(x, torch.randint(0, 10, (batch_size,)))
    mock_loader = DataLoader(mock_dataset, batch_size=8)
    
    metrics = model.analyze_synchronization(mock_loader)
    print("\nSynchronization Analysis:")
    print(f"  Temporal Synchronization: {metrics['temporal_synchronization']:.4f}")
    print(f"  Geometric Divergence: {metrics['geometric_divergence']:.4f}")
    print(f"  Parallel Processing Score: {metrics['parallel_score']:.4f}")
    
    return model

# %% [markdown]
# ## 3. Training Comparison

# %% Training comparison
def example_training_comparison():
    """Example 3: Compare PGN vs Hierarchical on MNIST"""
    print("\n" + "="*60)
    print("Example 3: Training Comparison")
    print("="*60)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create small subset for quick demo
    train_subset = torch.utils.data.Subset(train_dataset, range(1000))
    test_subset = torch.utils.data.Subset(test_dataset, range(200))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Initialize models
    pgn = ParallelGeometricNetwork(
        input_dim=784,
        hidden_dim=64,  # Smaller for quick demo
        output_dim=10,
        num_branches=4
    )
    
    hierarchical = HierarchicalNetwork(
        input_dim=784,
        hidden_dim=64,
        output_dim=10,
        num_layers=4
    )
    
    print(f"PGN parameters: {sum(p.numel() for p in pgn.parameters())}")
    print(f"Hierarchical parameters: {sum(p.numel() for p in hierarchical.parameters())}")
    
    # Train models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nTraining on {device}...")
    
    # Train PGN
    pgn_trainer = Trainer(pgn, device=device)
    pgn_history = pgn_trainer.train(
        train_loader, 
        test_loader, 
        num_epochs=3,  # Quick demo
        verbose=True,
        analyze_freq=2
    )
    
    # Train Hierarchical
    hier_trainer = Trainer(hierarchical, device=device)
    hier_history = hier_trainer.train(
        train_loader,
        test_loader,
        num_epochs=3,
        verbose=True
    )
    
    # Compare results
    print("\n" + "-"*40)
    print("Final Results:")
    print(f"PGN Test Accuracy: {pgn_history['val_acc'][-1]:.2f}%")
    print(f"Hierarchical Test Accuracy: {hier_history['val_acc'][-1]:.2f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].plot(pgn_history['train_loss'], 'o-', label='PGN')
    axes[0].plot(hier_history['train_loss'], 's-', label='Hierarchical')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(pgn_history['val_acc'], 'o-', label='PGN')
    axes[1].plot(hier_history['val_acc'], 's-', label='Hierarchical')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy (%)')
    axes[1].set_title('Validation Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pgn, hierarchical

# %% [markdown]
# ## 4. Sequential MNIST Example

# %% Sequential MNIST
def example_sequential_mnist():
    """Example 4: Sequential MNIST with Temporal PGN"""
    print("\n" + "="*60)
    print("Example 4: Sequential MNIST")
    print("="*60)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)
    
    # Convert to sequential
    seq_train = SequentialMNIST(mnist_train, sequence_type='row')
    seq_test = SequentialMNIST(mnist_test, sequence_type='row')
    
    # Create small subset for demo
    train_subset = torch.utils.data.Subset(seq_train, range(500))
    test_subset = torch.utils.data.Subset(seq_test, range(100))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Create models
    temporal_pgn = TemporalPGN(
        input_dim=28,
        hidden_dim=64,
        output_dim=10,
        num_branches=4
    )
    
    hierarchical_rnn = HierarchicalRNN(
        input_dim=28,
        hidden_dim=64,
        output_dim=10,
        num_layers=2
    )
    
    print(f"Temporal PGN parameters: {sum(p.numel() for p in temporal_pgn.parameters())}")
    print(f"Hierarchical RNN parameters: {sum(p.numel() for p in hierarchical_rnn.parameters())}")
    
    # Quick training demo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temporal_pgn = temporal_pgn.to(device)
    hierarchical_rnn = hierarchical_rnn.to(device)
    
    criterion = nn.CrossEntropyLoss()
    pgn_optimizer = torch.optim.Adam(temporal_pgn.parameters())
    rnn_optimizer = torch.optim.Adam(hierarchical_rnn.parameters())
    
    print(f"\nTraining for 2 epochs (demo)...")
    
    for epoch in range(2):
        # Train
        temporal_pgn.train()
        hierarchical_rnn.train()
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Train PGN
            pgn_optimizer.zero_grad()
            outputs = temporal_pgn(data)
            loss = criterion(outputs, labels)
            loss.backward()
            pgn_optimizer.step()
            
            # Train RNN
            rnn_optimizer.zero_grad()
            outputs = hierarchical_rnn(data)
            loss = criterion(outputs, labels)
            loss.backward()
            rnn_optimizer.step()
        
        # Evaluate
        temporal_pgn.eval()
        hierarchical_rnn.eval()
        
        pgn_correct = 0
        rnn_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                
                # PGN
                outputs = temporal_pgn(data)
                _, predicted = outputs.max(1)
                pgn_correct += (predicted == labels).sum().item()
                
                # RNN
                outputs = hierarchical_rnn(data)
                _, predicted = outputs.max(1)
                rnn_correct += (predicted == labels).sum().item()
                
                total += labels.size(0)
        
        pgn_acc = 100 * pgn_correct / total
        rnn_acc = 100 * rnn_correct / total
        
        print(f"Epoch {epoch+1}: PGN={pgn_acc:.1f}%, RNN={rnn_acc:.1f}%")
    
    # Analyze temporal PGN
    metrics = temporal_pgn.analyze_synchronization(test_loader, device)
    print(f"\nTemporal PGN Analysis:")
    print(f"  Temporal Sync: {metrics['temporal_synchronization']:.3f}")
    print(f"  Geometric Divergence: {metrics['geometric_divergence']:.3f}")
    print(f"  Parallel Score: {metrics['parallel_score']:.3f}")
    
    return temporal_pgn, hierarchical_rnn

# %% [markdown]
# ## 5. Advanced Features

# %% Advanced features
def example_advanced_features():
    """Example 5: Advanced PGN features"""
    print("\n" + "="*60)
    print("Example 5: Advanced Features")
    print("="*60)
    
    # Create model
    model = ParallelGeometricNetwork(
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        num_branches=4,
        dropout_rate=0.1  # With dropout
    )
    
    print("1. Branch Freezing/Unfreezing")
    print("-" * 30)
    
    # Freeze specific branches
    model.freeze_branches([0, 1])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"After freezing branches 0,1: {trainable_params} trainable params")
    
    # Unfreeze all
    model.unfreeze_branches()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"After unfreezing all: {trainable_params} trainable params")
    
    print("\n2. Branch-wise Analysis")
    print("-" * 30)
    
    x = torch.randn(16, 784)
    branch_reps = model.get_branch_representations(x)
    
    for i, rep in enumerate(branch_reps):
        print(f"Branch {i+1} representations:")
        for key, val in rep.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, mean={val.mean().item():.3f}")
    
    print("\n3. Different Initialization Strategies")
    print("-" * 30)
    
    # Custom geometric biases
    custom_model = ParallelGeometricNetwork(
        input_dim=100,
        hidden_dim=50,
        output_dim=10,
        num_branches=6,
        geometric_biases=['sparse', 'sparse', 'orthogonal', 
                         'orthogonal', 'gaussian', 'xavier']
    )
    
    print(f"Created model with custom bias distribution")
    print(f"Biases: 2×sparse, 2×orthogonal, 1×gaussian, 1×xavier")
    
    return model

# %% [markdown]
# ## Run All Examples

# %% Run all examples
if __name__ == "__main__":
    print("PGN Tutorial - Running All Examples")
    print("="*60)
    
    # Run all examples
    try:
        # Example 1: Basic PGN
        basic_model = example_basic_pgn()
        
        # Example 2: Temporal PGN
        temporal_model = example_temporal_pgn()
        
        # Example 3: Training comparison (small dataset)
        pgn_trained, hier_trained = example_training_comparison()
        
        # Example 4: Sequential MNIST
        temporal_pgn, hierarchical_rnn = example_sequential_mnist()
        
        # Example 5: Advanced features
        advanced_model = example_advanced_features()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()