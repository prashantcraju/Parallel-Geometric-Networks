"""
Enhanced PGN for Temporal Tasks
Training script for Sequential MNIST and other temporal benchmarks
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgn.models.temporal_pgn import TemporalPGN
from pgn.models.hierarchical_rnn import HierarchicalRNN
from pgn.datasets.sequential_mnist import SequentialMNIST, PermutedMNIST
from pgn.utils.visualization import plot_training_curves


def train_temporal_models(train_loader, test_loader, 
                         input_dim=28, num_epochs=10, device='cpu'):
    """
    Train and compare Temporal PGN vs Hierarchical RNN
    """
    # Initialize models
    pgn = TemporalPGN(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=10,
        num_branches=4,
        use_synchronized_bn=True,
        rnn_type='lstm'
    ).to(device)
    
    hierarchical = HierarchicalRNN(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=10,
        num_layers=2,
        rnn_type='lstm'
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    pgn_optimizer = torch.optim.Adam(pgn.parameters(), lr=0.001)
    hier_optimizer = torch.optim.Adam(hierarchical.parameters(), lr=0.001)
    
    pgn_accs = []
    hier_accs = []
    pgn_losses = []
    hier_losses = []
    
    print("Training on Sequential MNIST...")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Train PGN
        pgn.train()
        pgn_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - PGN')
        
        for data, labels in train_bar:
            data, labels = data.to(device), labels.to(device)
            
            pgn_optimizer.zero_grad()
            outputs = pgn(data)
            loss = criterion(outputs, labels)
            loss.backward()
            pgn_optimizer.step()
            pgn_loss += loss.item()
            
            train_bar.set_postfix({'loss': loss.item()})
        
        # Train Hierarchical
        hierarchical.train()
        hier_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Hierarchical')
        
        for data, labels in train_bar:
            data, labels = data.to(device), labels.to(device)
            
            hier_optimizer.zero_grad()
            outputs = hierarchical(data)
            loss = criterion(outputs, labels)
            loss.backward()
            hier_optimizer.step()
            hier_loss += loss.item()
            
            train_bar.set_postfix({'loss': loss.item()})
        
        # Evaluate
        pgn.eval()
        hierarchical.eval()
        
        pgn_correct = 0
        hier_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                
                # PGN accuracy
                outputs = pgn(data)
                _, predicted = torch.max(outputs.data, 1)
                pgn_correct += (predicted == labels).sum().item()
                
                # Hierarchical accuracy
                outputs = hierarchical(data)
                _, predicted = torch.max(outputs.data, 1)
                hier_correct += (predicted == labels).sum().item()
                
                total += labels.size(0)
        
        pgn_acc = 100 * pgn_correct / total
        hier_acc = 100 * hier_correct / total
        avg_pgn_loss = pgn_loss / len(train_loader)
        avg_hier_loss = hier_loss / len(train_loader)
        
        pgn_accs.append(pgn_acc)
        hier_accs.append(hier_acc)
        pgn_losses.append(avg_pgn_loss)
        hier_losses.append(avg_hier_loss)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Temporal PGN - Loss: {avg_pgn_loss:.4f}, Acc: {pgn_acc:.2f}%')
        print(f'  Hierarchical - Loss: {avg_hier_loss:.4f}, Acc: {hier_acc:.2f}%')
        print(f'  PGN Advantage: {pgn_acc - hier_acc:+.2f}%\n')
    
    # Analyze PGN synchronization
    print("\nAnalyzing PGN Synchronization...")
    metrics = pgn.analyze_synchronization(test_loader, device)
    print(f"  Temporal Synchronization: {metrics['temporal_synchronization']:.4f}")
    print(f"  Geometric Divergence: {metrics['geometric_divergence']:.4f}")
    print(f"  Parallel Processing Score: {metrics['parallel_score']:.4f}")
    
    return pgn, hierarchical, pgn_accs, hier_accs, pgn_losses, hier_losses, metrics


def visualize_temporal_dynamics(pgn, test_loader, device='cpu', save_path=None):
    """
    Visualize temporal dynamics and branch divergence
    """
    pgn.eval()
    
    # Get a batch of data
    data, labels = next(iter(test_loader))
    data = data.to(device)
    
    with torch.no_grad():
        _, temporal_features, branch_outputs = pgn(data[:8], return_dynamics=True)
    
    # Convert to numpy
    temporal_features = temporal_features.cpu().numpy()
    branch_outputs = [b.cpu().numpy() for b in branch_outputs]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Temporal dynamics (shared across branches)
    ax = axes[0, 0]
    for i in range(min(3, temporal_features.shape[0])):
        ax.plot(temporal_features[i, :, :5].T, alpha=0.7)
    ax.set_title('Shared Temporal Dynamics\n(Same for all branches)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Activation')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Branch divergence
    ax = axes[0, 1]
    branch_distances = []
    pair_labels = []
    
    for i in range(pgn.num_branches):
        for j in range(i+1, pgn.num_branches):
            dist = np.linalg.norm(branch_outputs[i] - branch_outputs[j], axis=1).mean()
            branch_distances.append(dist)
            pair_labels.append(f'{i+1}-{j+1}')
    
    bars = ax.bar(range(len(branch_distances)), branch_distances, color='steelblue')
    ax.set_title('Geometric Distances Between Branches')
    ax.set_xlabel('Branch Pair')
    ax.set_ylabel('L2 Distance')
    ax.set_xticks(range(len(branch_distances)))
    ax.set_xticklabels(pair_labels)
    
    # Add value labels on bars
    for bar, val in zip(bars, branch_distances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Branch representations (PCA)
    ax = axes[1, 0]
    
    pca = PCA(n_components=2)
    colors = plt.cm.Set1(np.linspace(0, 1, pgn.num_branches))
    
    for i, (branch_out, color) in enumerate(zip(branch_outputs, colors)):
        if branch_out.shape[0] > 2:  # Need at least 3 samples for PCA
            reduced = pca.fit_transform(branch_out)
            ax.scatter(reduced[:, 0], reduced[:, 1], 
                      label=f'Branch {i+1}', alpha=0.7, color=color, s=50)
    
    ax.set_title('Branch Representations (PCA)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Synchronization matrix
    ax = axes[1, 1]
    
    # Create synchronization matrix
    sync_matrix = np.ones((pgn.num_branches, pgn.num_branches))
    
    # Geometric divergence matrix
    div_matrix = np.zeros((pgn.num_branches, pgn.num_branches))
    for i in range(pgn.num_branches):
        for j in range(pgn.num_branches):
            if i != j:
                div_matrix[i, j] = np.linalg.norm(
                    branch_outputs[i].mean(0) - branch_outputs[j].mean(0)
                )
    
    if div_matrix.max() > 0:
        div_matrix = div_matrix / div_matrix.max()  # Normalize
    
    # Combined visualization
    combined = sync_matrix - 0.5 * div_matrix
    
    im = ax.imshow(combined, cmap='RdBu_r', vmin=0, vmax=1)
    ax.set_title('Synchronization (red) vs Divergence (blue)')
    ax.set_xlabel('Branch')
    ax.set_ylabel('Branch')
    ax.set_xticks(range(pgn.num_branches))
    ax.set_yticks(range(pgn.num_branches))
    ax.set_xticklabels([f'B{i+1}' for i in range(pgn.num_branches)])
    ax.set_yticklabels([f'B{i+1}' for i in range(pgn.num_branches)])
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_figure(pgn_accs, hier_accs, metrics, save_path=None):
    """
    Create publication-ready figure for NeurIPS
    """
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    
    fig = plt.figure(figsize=(15, 5))
    
    # Panel 1: Performance comparison
    ax1 = plt.subplot(131)
    epochs = range(1, len(pgn_accs) + 1)
    ax1.plot(epochs, pgn_accs, 'o-', label='Temporal PGN', 
             linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(epochs, hier_accs, 's-', label='Hierarchical RNN', 
             linewidth=2, markersize=8, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('A. Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(min(pgn_accs), min(hier_accs)) - 2, 
                  max(max(pgn_accs), max(hier_accs)) + 2])
    
    # Panel 2: Architecture principle
    ax2 = plt.subplot(132)
    
    # Bar plot showing the two key metrics
    metrics_names = ['Temporal\nSync', 'Geometric\nDivergence', 'Parallel\nScore']
    metrics_values = [
        metrics['temporal_synchronization'],
        metrics['geometric_divergence'],
        metrics['parallel_score']
    ]
    
    bars = ax2.bar(metrics_names, metrics_values, 
                   color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('B. PGN Characteristics', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(metrics_values) * 1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    
    # Panel 3: Advantage over epochs
    ax3 = plt.subplot(133)
    advantage = [p - h for p, h in zip(pgn_accs, hier_accs)]
    ax3.plot(epochs, advantage, 'go-', linewidth=2, markersize=8)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.fill_between(epochs, 0, advantage, where=[a > 0 for a in advantage],
                     color='green', alpha=0.2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('PGN Advantage (%)', fontsize=12)
    ax3.set_title('C. Relative Performance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add annotation for final advantage
    final_advantage = advantage[-1]
    if abs(final_advantage) > 0.5:  # Only annotate if significant
        ax3.annotate(f'Final: {final_advantage:+.2f}%',
                    xy=(len(advantage), final_advantage),
                    xytext=(len(advantage)-2, final_advantage + 1),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=11, color='green')
    
    plt.suptitle('Parallel Geometric Networks: Synchronized Time, Divergent Space',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load MNIST
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(args.data_dir, train=False, transform=transform)
    
    # Convert to Sequential MNIST
    print(f"Converting to Sequential MNIST (type: {args.sequence_type})...")
    
    if args.permuted:
        seq_train = PermutedMNIST(mnist_train, num_permutations=1)
        seq_test = PermutedMNIST(mnist_test, num_permutations=1)
    else:
        seq_train = SequentialMNIST(mnist_train, sequence_type=args.sequence_type)
        seq_test = SequentialMNIST(mnist_test, sequence_type=args.sequence_type)
    
    # Create dataloaders
    train_loader = DataLoader(seq_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(seq_test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Determine input dimension based on sequence type
    input_dim = 28 if args.sequence_type in ['row', 'column'] else 1
    
    # Train models
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    pgn, hierarchical, pgn_accs, hier_accs, pgn_losses, hier_losses, metrics = train_temporal_models(
        train_loader, test_loader, 
        input_dim=input_dim, 
        num_epochs=args.epochs, 
        device=device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save models if requested
    if args.save_models:
        torch.save(pgn.state_dict(), 
                   os.path.join(args.output_dir, 'temporal_pgn.pth'))
        torch.save(hierarchical.state_dict(), 
                   os.path.join(args.output_dir, 'hierarchical_rnn.pth'))
        print(f"\nModels saved to {args.output_dir}")
    
    # Create visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        
        # Temporal dynamics visualization
        fig1 = visualize_temporal_dynamics(
            pgn, test_loader, device,
            save_path=os.path.join(args.output_dir, 'temporal_dynamics.png')
        )
        
        # figure
        fig2 = create_figure(
            pgn_accs, hier_accs, metrics,
            save_path=os.path.join(args.output_dir, 'neurips_results.png')
        )
        
        # Training curves
        history = {
            'PGN Loss': pgn_losses,
            'Hierarchical Loss': hier_losses,
            'PGN Accuracy': pgn_accs,
            'Hierarchical Accuracy': hier_accs
        }
        
        fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        axes[0].plot(pgn_losses, 'o-', label='PGN', color='#2E86AB')
        axes[0].plot(hier_losses, 's-', label='Hierarchical', color='#A23B72')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(pgn_accs, 'o-', label='PGN', color='#2E86AB')
        axes[1].plot(hier_accs, 's-', label='Hierarchical', color='#A23B72')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Test Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Training History')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=150)
        
        if not args.no_show:
            plt.show()
        
        print(f"Visualizations saved to {args.output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Final PGN Accuracy: {pgn_accs[-1]:.2f}%")
    print(f"Final Hierarchical Accuracy: {hier_accs[-1]:.2f}%")
    print(f"PGN Advantage: {pgn_accs[-1] - hier_accs[-1]:+.2f}%")
    print(f"\nPGN Characteristics:")
    print(f"  Temporal Synchronization: {metrics['temporal_synchronization']:.4f}")
    print(f"  Geometric Divergence: {metrics['geometric_divergence']:.4f}")
    print(f"  Parallel Processing Score: {metrics['parallel_score']:.4f}")
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Temporal PGN on Sequential MNIST')
    
    # Model arguments
    parser.add_argument('--sequence_type', type=str, default='row',
                        choices=['row', 'column', 'spiral', 'random'],
                        help='Type of sequence generation')
    parser.add_argument('--permuted', action='store_true',
                        help='Use permuted MNIST instead')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for dataset')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs_temporal',
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    main(args)