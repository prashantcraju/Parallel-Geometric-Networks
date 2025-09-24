# pgn/__init__.py
"""
Parallel Geometric Networks (PGN)
Bio-inspired architecture with synchronized temporal dynamics 
and divergent geometric representations
"""

from .models.pgn import ParallelGeometricNetwork
from .models.temporal_pgn import TemporalPGN
from .models.hierarchical import HierarchicalNetwork
from .models.hierarchical_rnn import HierarchicalRNN
from .models.geometric_branch import GeometricBranch
from .layers.sync_batchnorm import SynchronizedBatchNorm
from .training.trainer import Trainer
from .datasets.sequential_mnist import SequentialMNIST, PermutedMNIST

__version__ = '0.2.0'

__all__ = [
    'ParallelGeometricNetwork',
    'TemporalPGN',
    'HierarchicalNetwork',
    'HierarchicalRNN',
    'GeometricBranch',
    'SynchronizedBatchNorm',
    'Trainer',
    'SequentialMNIST',
    'PermutedMNIST',
]
