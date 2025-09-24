"""
Visualization utilities for PGN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Optional, List, Tuple, Any
from torch.utils.data import DataLoader


def visualize_branch_representations(
    model: Any,
    data_loader: DataLoader,
    method: str = 'tsne',
    max_samples: int = 1000,
    layer_name: str = 'layer2',
    device: str = 'cpu',
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Visualize the geometric divergence of branch representations
    
    Args:
        model: PGN model
        data_loader: DataLoader for data
        method: Dimensionality reduction method ('tsne' or 'pca')
        max_samples: Maximum number of samples to visualize
        layer_name: Which layer to visualize
        device: Device to use
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    model.eval()
    
    all_branch_reps = [[] for _ in range(model.num_branches)]
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Flatten if needed
            if len(data.shape) > 2:
                data = data.view(data.size(0), -1)
            
            # Get branch representations
            branch_reps = model.get_branch_representations(data)
            
            for i in range(model.num_branches):
                rep = branch_reps[i][layer_name]
                all_branch_reps[i].append(rep.cpu().numpy())
            
            all_labels.append(labels.numpy())
            
            total_samples += len(labels)
            if total_samples >= max_samples:
                break
    
    # Concatenate all batches
    for i in range(model.num_branches):
        all_branch_reps[i] = np.vstack(all_branch_reps[i])[:max_samples]
    all_labels = np.concatenate(all_labels)[:max_samples]
    
    # Set figure size
    if figsize is None:
        figsize = (4 * model.num_branches, 4)
    
    # Create visualizations
    fig, axes = plt.subplots(1, model.num_branches, figsize=figsize)
    
    if model.num_branches == 1:
        axes = [axes]
    
    for i, (reps, ax) in enumerate(zip(all_branch_reps, axes)):
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Flatten representations if needed
        if len(reps.shape) > 2:
            reps = reps.reshape(reps.shape[0], -1)
        
        embedded = reducer.fit_transform(reps)
        
        # Plot
        scatter = ax.scatter(
            embedded[:, 0], 
            embedded[:, 1], 
            c=all_labels, 
            cmap='tab10', 
            alpha=0.6,
            s=10
        )
        
        # Get branch info
        branch_bias = model.branches[i].geometric_bias
        ax.set_title(f'Branch {i+1}\n({branch_bias})')
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.colorbar(scatter, ax=axes)
    plt.suptitle(f'Geometric Divergence Across Parallel Branches\n{layer_name}')
    plt.tight_layout()
    
    return fig


def plot_training_curves(
    history: dict,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot training curves
    
    Args:
        history: Dictionary containing training metrics
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train', marker='o')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Validation', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot additional metrics
    plotted_additional = False
    for key in history.keys():
        if key not in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
            axes[2].plot(history[key], label=key, marker='o')
            plotted_additional = True
    
    if plotted_additional:
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Value')
        axes[2].set_title('Additional Metrics')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def visualize_attention_maps(
    representations: List[torch.Tensor],
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Visualize attention-like patterns in branch representations
    
    Args:
        representations: List of representation tensors
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    num_branches = len(representations)
    fig, axes = plt.subplots(1, num_branches, figsize=figsize)
    
    if num_branches == 1:
        axes = [axes]
    
    for i, (rep, ax) in enumerate(zip(representations, axes)):
        # Compute self-attention-like scores
        rep_flat = rep.flatten(1)
        attention = torch.matmul(rep_flat, rep_flat.T)
        attention = F.softmax(attention, dim=-1)
        
        # Plot heatmap
        im = ax.imshow(attention.cpu().numpy(), cmap='hot', aspect='auto')
        ax.set_title(f'Branch {i+1} Similarity')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sample Index')
        
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Inter-sample Similarity Patterns')
    plt.tight_layout()
    
    return fig


def plot_comparison_results(
    pgn_metrics: dict,
    baseline_metrics: dict,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot comparison between PGN and baseline model
    
    Args:
        pgn_metrics: PGN performance metrics
        baseline_metrics: Baseline performance metrics
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy comparison
    if 'accuracy' in pgn_metrics and 'accuracy' in baseline_metrics:
        models = ['PGN', 'Hierarchical']
        accuracies = [pgn_metrics['accuracy'], baseline_metrics['accuracy']]
        
        bars = axes[0].bar(models, accuracies, color=['#2E86AB', '#A23B72'])
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylim([min(accuracies) * 0.9, min(100, max(accuracies) * 1.1)])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}%', ha='center', va='bottom')
    
    # Training time or other metrics
    metrics_to_plot = {}
    for key in pgn_metrics.keys():
        if key != 'accuracy' and key in baseline_metrics:
            metrics_to_plot[key] = (pgn_metrics[key], baseline_metrics[key])
    
    if metrics_to_plot:
        metric_names = list(metrics_to_plot.keys())
        pgn_values = [metrics_to_plot[m][0] for m in metric_names]
        baseline_values = [metrics_to_plot[m][1] for m in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[1].bar(x - width/2, pgn_values, width, label='PGN', color='#2E86AB')
        axes[1].bar(x + width/2, baseline_values, width, label='Hierarchical', color='#A23B72')
        
        axes[1].set_xlabel('Metrics')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Additional Metrics Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1].legend()
    
    plt.suptitle('PGN vs Hierarchical Network')
    plt.tight_layout()
    
    return fig