"""
Comprehensive model comparison script
Compares PGN variants against baseline models on multiple datasets
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgn.models.pgn import ParallelGeometricNetwork
from pgn.models.temporal_pgn import TemporalPGN
from pgn.models.hierarchical import HierarchicalNetwork
from pgn.models.hierarchical_rnn import HierarchicalRNN
from pgn.datasets.sequential_mnist import SequentialMNIST, PermutedMNIST
from pgn.training.trainer import Trainer
from pgn.utils.analysis import analyze_representations
from pgn.utils.visualization import plot_comparison_results


class ModelComparator:
    """
    Comprehensive model comparison framework
    """
    
    def __init__(self, 
                 device: str = 'cpu',
                 results_dir: str = './comparison_results',
                 seed: int = 42):
        """
        Initialize comparator
        
        Args:
            device: Device to use for training
            results_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.device = device
        self.results_dir = results_dir
        self.seed = seed
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Store results
        self.results = {
            'models': {},
            'metrics': {},
            'timing': {},
            'analysis': {}
        }
    
    def load_dataset(self, 
                    dataset_name: str,
                    batch_size: int = 64,
                    subset_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load dataset for comparison
        
        Args:
            dataset_name: Name of dataset ('mnist', 'sequential_mnist', 'permuted_mnist', 'cifar10')
            batch_size: Batch size for data loaders
            subset_size: Use subset of data for quick testing
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        print(f"Loading {dataset_name} dataset...")
        
        if dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            
            # Split train into train/val
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
        elif dataset_name == 'sequential_mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
            mnist_test = datasets.MNIST('./data', train=False, transform=transform)
            
            # Convert to sequential
            train_dataset = SequentialMNIST(mnist_train, sequence_type='row')
            test_dataset = SequentialMNIST(mnist_test, sequence_type='row')
            
            # Split train
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
        elif dataset_name == 'permuted_mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
            mnist_test = datasets.MNIST('./data', train=False, transform=transform)
            
            # Convert to permuted
            train_dataset = PermutedMNIST(mnist_train, num_permutations=1)
            test_dataset = PermutedMNIST(mnist_test, num_permutations=1)
            
            # Split train
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
        elif dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
            
            # Split train
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Create subsets if requested
        if subset_size:
            train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
            val_dataset = Subset(val_dataset, range(min(subset_size // 5, len(val_dataset))))
            test_dataset = Subset(test_dataset, range(min(subset_size // 5, len(test_dataset))))
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def create_models(self, 
                     model_configs: Dict[str, Dict],
                     input_shape: Tuple[int, ...]) -> Dict[str, nn.Module]:
        """
        Create models for comparison
        
        Args:
            model_configs: Configuration for each model
            input_shape: Shape of input data
            
        Returns:
            Dictionary of models
        """
        models = {}
        
        for name, config in model_configs.items():
            print(f"Creating {name}...")
            
            if config['type'] == 'pgn':
                if len(input_shape) == 1:  # Flat input
                    model = ParallelGeometricNetwork(
                        input_dim=input_shape[0],
                        **config.get('params', {})
                    )
                else:
                    raise ValueError("Use TemporalPGN for sequential data")
                    
            elif config['type'] == 'temporal_pgn':
                if len(input_shape) == 2:  # Sequential input
                    model = TemporalPGN(
                        input_dim=input_shape[1],
                        **config.get('params', {})
                    )
                else:
                    raise ValueError("TemporalPGN requires sequential data")
                    
            elif config['type'] == 'hierarchical':
                if len(input_shape) == 1:
                    model = HierarchicalNetwork(
                        input_dim=input_shape[0],
                        **config.get('params', {})
                    )
                else:
                    raise ValueError("Use HierarchicalRNN for sequential data")
                    
            elif config['type'] == 'hierarchical_rnn':
                if len(input_shape) == 2:
                    model = HierarchicalRNN(
                        input_dim=input_shape[1],
                        **config.get('params', {})
                    )
                else:
                    raise ValueError("HierarchicalRNN requires sequential data")
                    
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            models[name] = model.to(self.device)
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {param_count:,} parameters")
            self.results['models'][name] = {
                'type': config['type'],
                'parameters': param_count
            }
        
        return models
    
    def train_model(self,
                   model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   num_epochs: int = 10,
                   learning_rate: float = 0.001) -> Dict[str, List]:
        """
        Train a single model
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        trainer = Trainer(
            model=model,
            device=self.device,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            verbose=False,
            analyze_freq=None  # Skip analysis during training for speed
        )
        
        return history
    
    def evaluate_model(self,
                      model: nn.Module,
                      test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Flatten if needed
                if len(data.shape) > 2:
                    if hasattr(model, 'temporal_encoder'):  # Temporal model
                        pass  # Keep shape
                    else:
                        data = data.view(data.size(0), -1)
                        
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def analyze_model(self,
                     model: nn.Module,
                     test_loader: DataLoader,
                     model_name: str) -> Dict[str, Any]:
        """
        Analyze model properties
        
        Args:
            model: Model to analyze
            test_loader: Test data loader
            model_name: Name of the model
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        # Get sample batch
        data, _ = next(iter(test_loader))
        data = data.to(self.device)
        
        # Flatten if needed
        if len(data.shape) > 2 and not hasattr(model, 'temporal_encoder'):
            data = data.view(data.size(0), -1)
        
        # PGN-specific analysis
        if hasattr(model, 'analyze_representations'):
            rep_analysis = analyze_representations(model, data[:32])
            analysis['representation_analysis'] = rep_analysis
        
        # Temporal PGN analysis
        if hasattr(model, 'analyze_synchronization'):
            sync_analysis = model.analyze_synchronization(test_loader, self.device)
            analysis['synchronization_analysis'] = sync_analysis
        
        # Measure inference time
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):
                _ = model(data[:32])
            inference_time = (time.time() - start_time) / 10
        
        analysis['inference_time'] = inference_time
        
        # Memory usage
        if torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = model(data[:32])
            memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            analysis['memory_mb'] = memory_used
        
        return analysis
    
    def run_comparison(self,
                      dataset_name: str,
                      model_configs: Dict[str, Dict],
                      num_epochs: int = 10,
                      batch_size: int = 64,
                      subset_size: int = None) -> Dict[str, Any]:
        """
        Run complete comparison
        
        Args:
            dataset_name: Dataset to use
            model_configs: Model configurations
            num_epochs: Training epochs
            batch_size: Batch size
            subset_size: Use subset for quick testing
            
        Returns:
            Comparison results
        """
        print("="*60)
        print(f"Running comparison on {dataset_name}")
        print("="*60)
        
        # Load data
        train_loader, val_loader, test_loader = self.load_dataset(
            dataset_name, batch_size, subset_size
        )
        
        # Determine input shape
        sample_data, _ = next(iter(train_loader))
        if len(sample_data.shape) == 4:  # Image data
            input_shape = (sample_data.shape[1] * sample_data.shape[2] * sample_data.shape[3],)
        elif len(sample_data.shape) == 3:  # Sequential data
            input_shape = sample_data.shape[1:]  # (time, features)
        else:
            input_shape = sample_data.shape[1:]
        
        print(f"Input shape: {input_shape}")
        
        # Create models
        models = self.create_models(model_configs, input_shape)
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Training
            start_time = time.time()
            history = self.train_model(
                model, train_loader, val_loader, num_epochs
            )
            train_time = time.time() - start_time
            
            # Evaluation
            test_metrics = self.evaluate_model(model, test_loader)
            
            # Analysis
            analysis = self.analyze_model(model, test_loader, name)
            
            # Store results
            self.results['metrics'][name] = test_metrics
            self.results['timing'][name] = {
                'train_time': train_time,
                'inference_time': analysis.get('inference_time', None)
            }
            self.results['analysis'][name] = analysis
            
            print(f"  Test Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"  Training Time: {train_time:.2f}s")
            print(f"  Inference Time: {analysis.get('inference_time', 0)*1000:.2f}ms")
        
        return self.results
    
    def create_comparison_report(self, save_path: str = None):
        """
        Create comprehensive comparison report
        
        Args:
            save_path: Path to save report
        """
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f'comparison_{timestamp}.html')
        
        # Create pandas DataFrames
        metrics_df = pd.DataFrame(self.results['metrics']).T
        timing_df = pd.DataFrame(self.results['timing']).T
        models_df = pd.DataFrame(self.results['models']).T
        
        # Combine into single report
        report = f"""
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Model Comparison Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Model Information</h2>
            {models_df.to_html()}
            
            <h2>Performance Metrics</h2>
            {metrics_df.to_html()}
            
            <h2>Timing Information</h2>
            {timing_df.to_html()}
            
            <h2>Summary</h2>
            <ul>
                <li>Best Accuracy: {metrics_df['accuracy'].idxmax()} ({metrics_df['accuracy'].max():.2f}%)</li>
                <li>Fastest Training: {timing_df['train_time'].idxmin()} ({timing_df['train_time'].min():.2f}s)</li>
                <li>Fastest Inference: {timing_df['inference_time'].idxmin()} ({timing_df['inference_time'].min()*1000:.2f}ms)</li>
            </ul>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {save_path}")
        
        # Also save raw results as JSON
        json_path = save_path.replace('.html', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return save_path
    
    def plot_comparison(self, save_path: str = None):
        """
        Create comparison visualizations
        
        Args:
            save_path: Path to save plot
        """
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f'comparison_{timestamp}.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy comparison
        ax = axes[0, 0]
        models = list(self.results['metrics'].keys())
        accuracies = [self.results['metrics'][m]['accuracy'] for m in models]
        bars = ax.bar(models, accuracies, color=plt.cm.Set2(np.linspace(0, 1, len(models))))
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim([min(accuracies) * 0.95, min(100, max(accuracies) * 1.02)])
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{acc:.1f}%', ha='center', va='bottom')
        
        # Parameter count
        ax = axes[0, 1]
        param_counts = [self.results['models'][m]['parameters'] for m in models]
        bars = ax.bar(models, param_counts, color=plt.cm.Set2(np.linspace(0, 1, len(models))))
        ax.set_ylabel('Parameters')
        ax.set_title('Model Complexity')
        ax.set_yscale('log')
        
        for bar, count in zip(bars, param_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{count:,}', ha='center', va='bottom', rotation=45)
        
        # Training time
        ax = axes[1, 0]
        train_times = [self.results['timing'][m]['train_time'] for m in models]
        bars = ax.bar(models, train_times, color=plt.cm.Set2(np.linspace(0, 1, len(models))))
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Training Efficiency')
        
        for bar, time_val in zip(bars, train_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time_val:.1f}s', ha='center', va='bottom')
        
        # Inference time
        ax = axes[1, 1]
        inference_times = [self.results['timing'][m]['inference_time'] * 1000 for m in models]
        bars = ax.bar(models, inference_times, color=plt.cm.Set2(np.linspace(0, 1, len(models))))
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Speed')
        
        for bar, time_val in zip(bars, inference_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time_val:.2f}ms', ha='center', va='bottom')
        
        plt.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Comparison plot saved to: {save_path}")
        
        return save_path


def main(args):
    """Main comparison function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize comparator
    comparator = ModelComparator(
        device=device,
        results_dir=args.output_dir,
        seed=args.seed
    )
    
    # Define model configurations based on dataset
    if args.dataset in ['mnist', 'cifar10']:
        # Standard models for image classification
        model_configs = {
            'PGN-4': {
                'type': 'pgn',
                'params': {
                    'hidden_dim': args.hidden_dim,
                    'output_dim': 10,
                    'num_branches': 4,
                    'dropout_rate': args.dropout
                }
            },
            'PGN-8': {
                'type': 'pgn',
                'params': {
                    'hidden_dim': args.hidden_dim,
                    'output_dim': 10,
                    'num_branches': 8,
                    'dropout_rate': args.dropout
                }
            },
            'Hierarchical-4': {
                'type': 'hierarchical',
                'params': {
                    'hidden_dim': args.hidden_dim,
                    'output_dim': 10,
                    'num_layers': 4,
                    'dropout_rate': args.dropout
                }
            },
            'Hierarchical-6': {
                'type': 'hierarchical',
                'params': {
                    'hidden_dim': args.hidden_dim,
                    'output_dim': 10,
                    'num_layers': 6,
                    'dropout_rate': args.dropout
                }
            }
        }
    else:
        # Temporal models for sequential data
        model_configs = {
            'TemporalPGN-4': {
                'type': 'temporal_pgn',
                'params': {
                    'hidden_dim': args.hidden_dim,
                    'output_dim': 10,
                    'num_branches': 4,
                    'rnn_type': 'lstm'
                }
            },
            'TemporalPGN-8': {
                'type': 'temporal_pgn',
                'params': {
                    'hidden_dim': args.hidden_dim,
                    'output_dim': 10,
                    'num_branches': 8,
                    'rnn_type': 'lstm'
                }
            },
            'HierarchicalRNN-2': {
                'type': 'hierarchical_rnn',
                'params': {
                    'hidden_dim': args.hidden_dim,
                    'output_dim': 10,
                    'num_layers': 2,
                    'rnn_type': 'lstm'
                }
            },
            'HierarchicalRNN-3': {
                'type': 'hierarchical_rnn',
                'params': {
                    'hidden_dim': args.hidden_dim,
                    'output_dim': 10,
                    'num_layers': 3,
                    'rnn_type': 'lstm'
                }
            }
        }
    
    # Run comparison
    results = comparator.run_comparison(
        dataset_name=args.dataset,
        model_configs=model_configs,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        subset_size=args.subset_size
    )
    
    # Generate report and visualizations
    if args.generate_report:
        report_path = comparator.create_comparison_report()
        plot_path = comparator.plot_comparison()
        
        print("\n" + "="*60)
        print("Comparison Complete!")
        print("="*60)
        print(f"Report: {report_path}")
        print(f"Plot: {plot_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Find best model
    accuracies = {name: results['metrics'][name]['accuracy'] 
                  for name in results['metrics']}
    best_model = max(accuracies, key=accuracies.get)
    
    print(f"Best Model: {best_model}")
    print(f"Best Accuracy: {accuracies[best_model]:.2f}%")
    
    # Print all results
    print("\nAll Results:")
    for name in results['metrics']:
        print(f"  {name:20s}: {accuracies[name]:.2f}%")
    
    # Print PGN advantage if applicable
    pgn_models = [name for name in accuracies if 'PGN' in name]
    baseline_models = [name for name in accuracies if 'PGN' not in name]
    
    if pgn_models and baseline_models:
        best_pgn = max(pgn_models, key=lambda x: accuracies[x])
        best_baseline = max(baseline_models, key=lambda x: accuracies[x])
        advantage = accuracies[best_pgn] - accuracies[best_baseline]
        
        print(f"\nPGN Advantage: {advantage:+.2f}%")
        print(f"  Best PGN: {best_pgn} ({accuracies[best_pgn]:.2f}%)")
        print(f"  Best Baseline: {best_baseline} ({accuracies[best_baseline]:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare PGN models against baselines')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'sequential_mnist', 'permuted_mnist', 'cifar10'],
                        help='Dataset to use for comparison')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for models')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Use subset of data for quick testing')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                        help='Directory for outputs')
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate HTML report and plots')
    
    # Other arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    main(args)