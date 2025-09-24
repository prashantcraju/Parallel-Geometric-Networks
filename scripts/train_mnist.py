"""
Training script for PGN on MNIST
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgn.models.pgn import ParallelGeometricNetwork
from pgn.models.hierarchical import HierarchicalNetwork
from pgn.training.trainer import Trainer
from pgn.utils.visualization import (
    visualize_branch_representations,
    plot_training_curves,
    plot_comparison_results
)
from pgn.utils.analysis import analyze_representations


def load_mnist(batch_size: int = 64, data_dir: str = './data'):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, transform=transform
    )
    
    # Split train into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_mnist(args.batch_size, args.data_dir)
    
    # Initialize models
    print("Initializing models...")
    
    # PGN model
    pgn = ParallelGeometricNetwork(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        output_dim=10,
        num_branches=args.num_branches,
        geometric_biases=args.geometric_biases,
        use_synchronized_bn=not args.no_sync_bn,
        dropout_rate=args.dropout
    )
    
    # Hierarchical baseline
    hierarchical = HierarchicalNetwork(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        output_dim=10,
        num_layers=args.num_branches,  # Match depth
        dropout_rate=args.dropout
    )
    
    # Print model information
    pgn_params = sum(p.numel() for p in pgn.parameters())
    hier_params = sum(p.numel() for p in hierarchical.parameters())
    print(f"PGN parameters: {pgn_params:,}")
    print(f"Hierarchical parameters: {hier_params:,}")
    
    # Train PGN
    print("\n" + "="*60)
    print("Training Parallel Geometric Network...")
    print("="*60)
    
    pgn_trainer = Trainer(
        model=pgn,
        device=device,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(pgn.parameters(), lr=args.lr),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            torch.optim.Adam(pgn.parameters(), lr=args.lr), 
            patience=3
        ) if args.use_scheduler else None
    )
    
    pgn_history = pgn_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        verbose=not args.quiet,
        analyze_freq=5 if not args.no_analysis else None,
        early_stopping_patience=args.early_stopping
    )
    
    # Train Hierarchical
    print("\n" + "="*60)
    print("Training Hierarchical Network...")
    print("="*60)
    
    hier_trainer = Trainer(
        model=hierarchical,
        device=device,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(hierarchical.parameters(), lr=args.lr),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            torch.optim.Adam(hierarchical.parameters(), lr=args.lr),
            patience=3
        ) if args.use_scheduler else None
    )
    
    hier_history = hier_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        verbose=not args.quiet,
        early_stopping_patience=args.early_stopping
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    pgn_test_results = pgn_trainer.evaluate(test_loader)
    hier_test_results = hier_trainer.evaluate(test_loader)
    
    print(f"PGN Test Accuracy: {pgn_test_results['accuracy']:.2f}%")
    print(f"Hierarchical Test Accuracy: {hier_test_results['accuracy']:.2f}%")
    
    improvement = pgn_test_results['accuracy'] - hier_test_results['accuracy']
    print(f"PGN Advantage: {improvement:+.2f}%")
    
    # Final analysis
    if not args.no_analysis:
        print("\n" + "="*60)
        print("Final PGN Analysis")
        print("="*60)
        
        sample_data, _ = next(iter(test_loader))
        sample_data = sample_data.to(device).view(sample_data.size(0), -1)
        
        analysis = analyze_representations(pgn, sample_data)
        print(f"Mean Geometric Distance: {analysis['mean_geometric_distance']:.4f}")
        if analysis['mean_temporal_correlation'] is not None:
            print(f"Mean Temporal Correlation: {analysis['mean_temporal_correlation']:.4f}")
        print(f"Branch Specialization: {analysis['branch_specialization']:.4f}")
        print(f"Mean Effective Rank: {analysis['mean_effective_rank']:.2f}")
    
    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # Training curves
        fig1 = plot_training_curves(pgn_history, "PGN Training History")
        fig1.savefig(os.path.join(args.output_dir, 'pgn_training_curves.png'))
        
        fig2 = plot_training_curves(hier_history, "Hierarchical Training History")
        fig2.savefig(os.path.join(args.output_dir, 'hier_training_curves.png'))
        
        # Branch representations
        fig3 = visualize_branch_representations(
            pgn, test_loader, method='tsne', device=device
        )
        fig3.savefig(os.path.join(args.output_dir, 'branch_representations.png'))
        
        # Comparison
        fig4 = plot_comparison_results(pgn_test_results, hier_test_results)
        fig4.savefig(os.path.join(args.output_dir, 'model_comparison.png'))
        
        if not args.no_show:
            plt.show()
        
        print(f"Visualizations saved to {args.output_dir}")
    
    # Save models
    if args.save_models:
        torch.save(pgn.state_dict(), 
                   os.path.join(args.output_dir, 'pgn_model.pth'))
        torch.save(hierarchical.state_dict(), 
                   os.path.join(args.output_dir, 'hierarchical_model.pth'))
        print(f"Models saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PGN on MNIST')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for models')
    parser.add_argument('--num_branches', type=int, default=4,
                        help='Number of parallel branches')
    parser.add_argument('--geometric_biases', nargs='+', 
                        default=['sparse', 'orthogonal', 'gaussian', 'xavier'],
                        help='Geometric bias types for branches')
    parser.add_argument('--no_sync_bn', action='store_true',
                        help='Disable synchronized batch normalization')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=None,
                        help='Early stopping patience')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for dataset')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory for outputs')
    parser.add_argument('--save_models', action='store_true',
                        help='Save trained models')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not show plots')
    
    # Other arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--no_analysis', action='store_true',
                        help='Skip analysis during training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)