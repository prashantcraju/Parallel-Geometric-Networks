# Parallel Geometric Networks (PGN)

Bio-inspired neural architecture with synchronized temporal dynamics and divergent geometric representations.

##  Overview

PGN is a novel neural network architecture inspired by biological neural systems. It features:

- **Parallel Processing Branches**: Multiple specialized pathways process information simultaneously
- **Synchronized Temporal Dynamics**: Shared temporal statistics across branches via synchronized batch normalization
- **Divergent Geometric Representations**: Each branch develops unique geometric biases for different feature types
- **Temporal Extensions**: Specialized variants for sequential and time-series data
- **Improved Generalization**: Better performance compared to traditional hierarchical networks

##  Enhanced Temporal Capabilities (v0.2.0)

The enhanced version includes specialized temporal models:
- **TemporalPGN**: RNN-based PGN for sequential data
- **SequentialMNIST**: Dataset utilities for temporal experiments
- **Synchronization Analysis**: Metrics for temporal-geometric trade-offs

##  Architecture

The key components of PGN include:

1. **Geometric Branches**: Parallel processing units with different initialization strategies (sparse, orthogonal, gaussian, xavier)
2. **Synchronized BatchNorm**: Ensures temporal coherence while allowing geometric divergence
3. **Fusion Layer**: Combines branch outputs for final predictions

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pgn.git
cd pgn

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
import torch
from pgn import ParallelGeometricNetwork

# Create model
model = ParallelGeometricNetwork(
    input_dim=784,
    hidden_dim=128,
    output_dim=10,
    num_branches=4,
    geometric_biases=['sparse', 'orthogonal', 'gaussian', 'xavier'],
    use_synchronized_bn=True
)

# Forward pass
x = torch.randn(32, 784)  # batch_size=32, input_dim=784
output = model(x)  # shape: (32, 10)

# Get branch representations for analysis
representations = model.get_branch_representations(x)
```

### Training on MNIST

```bash
# Basic training
python scripts/train_mnist.py --epochs 20 --visualize

# With all features
python scripts/train_mnist.py \
    --epochs 30 \
    --num_branches 4 \
    --hidden_dim 256 \
    --lr 0.001 \
    --visualize \
    --save_models \
    --early_stopping 5
```

### Training Temporal Models

```bash
# Sequential MNIST with row-wise reading
python scripts/train_temporal.py \
    --sequence_type row \
    --epochs 10 \
    --visualize

# Permuted MNIST
python scripts/train_temporal.py \
    --permuted \
    --epochs 15 \
    --save_models

# Spiral reading pattern
python scripts/train_temporal.py \
    --sequence_type spiral \
    --batch_size 128 \
    --visualize
```

## Repository Structure

```
Parallel-Geometric-Networks/
├── pgn/                      # Core package
│   ├── layers/               # Custom layers
│   │   └── sync_batchnorm.py
│   ├── models/               # Model architectures
│   │   ├── geometric_branch.py
│   │   ├── pgn.py
│   │   ├── temporal_pgn.py  # NEW: Temporal variant
│   │   ├── hierarchical.py
│   │   └── hierarchical_rnn.py  # NEW: RNN baseline
│   ├── datasets/             # Dataset utilities
│   │   └── sequential_mnist.py  # NEW: Sequential datasets
│   ├── utils/                # Utilities
│   │   ├── analysis.py
│   │   └── visualization.py
│   └── training/             # Training utilities
│       └── trainer.py
├── scripts/                  # Training and evaluation scripts
│   ├── train_mnist.py
│   └── train_temporal.py    # NEW: Temporal training
├── configs/                  # Configuration files
├── examples/                 # Example notebooks
│   └── pgn_tutorial.py       # Complete tutorial
└── tests/                    # Unit tests
```

##  Key Features

### 1. Geometric Biases

Different initialization strategies for branches:
- **Sparse**: Local feature detection
- **Orthogonal**: Decorrelated representations
- **Gaussian**: Multi-scale features
- **Xavier**: Balanced initialization

### 2. Synchronized Batch Normalization

Maintains temporal coherence across branches while allowing geometric specialization:

```python
from pgn.layers import SynchronizedBatchNorm

sync_bn = SynchronizedBatchNorm(
    num_features=128,
    num_branches=4
)
```

### 3. Temporal Models for Sequential Data

Process sequential and time-series data with synchronized temporal dynamics:

```python
from pgn import TemporalPGN, SequentialMNIST

# Create temporal model
model = TemporalPGN(
    input_dim=28,      # Features per timestep
    hidden_dim=128,
    output_dim=10,
    num_branches=4,
    rnn_type='lstm'
)

# Process sequential data
x = torch.randn(32, 28, 28)  # (batch, time, features)
output = model(x)

# Analyze synchronization
metrics = model.analyze_synchronization(data_loader, device)
print(f"Temporal Sync: {metrics['temporal_synchronization']:.3f}")
print(f"Geometric Divergence: {metrics['geometric_divergence']:.3f}")
```

### 4. Analysis Tools

Comprehensive analysis of learned representations:

```python
from pgn.utils.analysis import analyze_representations

analysis = analyze_representations(model, data)
print(f"Geometric Distance: {analysis['mean_geometric_distance']:.4f}")
print(f"Temporal Correlation: {analysis['mean_temporal_correlation']:.4f}")
```

### 4. Visualization

Rich visualization tools for understanding model behavior:

```python
from pgn.utils.visualization import visualize_branch_representations

fig = visualize_branch_representations(
    model, 
    data_loader,
    method='tsne'  # or 'pca'
)
```

##  Performance

### Standard MNIST
- **PGN**: ~98.5% accuracy
- **Hierarchical Baseline**: ~97.8% accuracy
- **Improvement**: +0.7% with similar parameter count

### Sequential MNIST
- **Temporal PGN**: ~97.2% accuracy
- **Hierarchical RNN**: ~96.3% accuracy
- **Improvement**: +0.9% with better temporal-geometric trade-off

### Key Metrics
- **Temporal Synchronization**: 0.85-0.92 (higher is better)
- **Geometric Divergence**: 0.65-0.75 (higher is better)
- **Parallel Processing Score**: 0.55-0.69 (combined metric)

## 🔬 Research Applications

PGN is suitable for:
- Understanding parallel processing in neural networks
- Exploring geometric properties of learned representations
- Investigating synchronization mechanisms in deep learning
- Developing bio-inspired architectures

##  Configuration

### Model Configuration

```python
config = {
    'input_dim': 784,
    'hidden_dim': 256,
    'output_dim': 10,
    'num_branches': 4,
    'geometric_biases': ['sparse', 'orthogonal', 'gaussian', 'xavier'],
    'use_synchronized_bn': True,
    'dropout_rate': 0.1
}

model = ParallelGeometricNetwork(**config)
```

### Training Configuration

```python
from pgn.training import Trainer

trainer = Trainer(
    model=model,
    device='cuda',
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    analyze_freq=5  # Run analysis every 5 epochs
)
```

## Advanced Usage

### Custom Geometric Biases

```python
from pgn.models import GeometricBranch

class CustomBranch(GeometricBranch):
    def _initialize_geometry(self):
        # Implement custom initialization
        for layer in [self.fc1, self.fc2, self.fc3]:
            # Your custom initialization logic
            pass
```

### Fine-tuning Specific Branches

```python
# Freeze certain branches
model.freeze_branches([0, 1])  # Freeze first two branches

# Train only unfrozen branches
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters())
)
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- seaborn

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pgn2025,
  title = {Parallel Geometric Networks},
  author = {Prashant C. Raju},
  year = {2025},
  url = {https://github.com/prashantcraju/Parallel-Geometric-Networks}
}
```

##  Acknowledgments

This implementation is inspired by biological neural systems and recent advances in parallel processing architectures.
