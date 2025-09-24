"""
Analysis utilities for PGN
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any
from ..models.pgn import ParallelGeometricNetwork


def analyze_representations(
    model: ParallelGeometricNetwork, 
    x: torch.Tensor,
    layer_name: str = 'layer2'
) -> Dict[str, Any]:
    """
    Analyze geometric divergence and temporal synchrony in PGN
    
    Args:
        model: PGN model
        x: Input tensor
        layer_name: Which layer to analyze
        
    Returns:
        Dictionary containing analysis metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Get representations from each branch
        all_reps = model.get_branch_representations(x)
        
        # Compute geometric distances between branches
        geometric_distances = []
        for i in range(model.num_branches):
            for j in range(i+1, model.num_branches):
                # Compute cosine distance between branch representations
                rep_i = all_reps[i][layer_name].flatten(1)
                rep_j = all_reps[j][layer_name].flatten(1)
                
                cos_sim = F.cosine_similarity(rep_i, rep_j, dim=1).mean()
                geometric_distances.append(1 - cos_sim.item())
        
        # Compute temporal correlations (if using synchronized BN)
        temporal_correlations = []
        if model.use_synchronized_bn:
            # Check correlation of activations over batch dimension
            for i in range(model.num_branches):
                for j in range(i+1, model.num_branches):
                    rep_i = all_reps[i][layer_name]
                    rep_j = all_reps[j][layer_name]
                    
                    # Correlation across batch dimension (temporal)
                    if rep_i.shape[0] > 1:  # Need at least 2 samples
                        corr = torch.corrcoef(torch.stack([
                            rep_i.mean(dim=1).flatten(),
                            rep_j.mean(dim=1).flatten()
                        ]))[0, 1]
                        temporal_correlations.append(corr.item())
        
        # Compute branch specialization (variance of outputs)
        branch_outputs = []
        for rep in all_reps:
            branch_outputs.append(rep['output'])
        branch_outputs = torch.stack(branch_outputs)
        specialization = branch_outputs.var(dim=0).mean().item()
        
        # Compute effective rank of representations
        effective_ranks = []
        for rep in all_reps:
            h = rep[layer_name].flatten(1)
            _, s, _ = torch.svd(h)
            # Normalized singular values
            s_norm = s / s.sum()
            # Entropy-based effective rank
            entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
            effective_rank = torch.exp(entropy).item()
            effective_ranks.append(effective_rank)
        
        return {
            'geometric_distances': geometric_distances,
            'temporal_correlations': temporal_correlations,
            'mean_geometric_distance': np.mean(geometric_distances),
            'mean_temporal_correlation': np.mean(temporal_correlations) if temporal_correlations else None,
            'branch_specialization': specialization,
            'effective_ranks': effective_ranks,
            'mean_effective_rank': np.mean(effective_ranks)
        }


def compute_gradient_metrics(
    model: ParallelGeometricNetwork,
    loss: torch.Tensor
) -> Dict[str, float]:
    """
    Compute gradient-based metrics for analysis
    
    Args:
        model: PGN model
        loss: Loss tensor (must have grad)
        
    Returns:
        Dictionary of gradient metrics
    """
    # Compute gradients
    loss.backward(retain_graph=True)
    
    gradient_norms = []
    gradient_variances = []
    
    for i, branch in enumerate(model.branches):
        branch_grad_norms = []
        
        for param in branch.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                branch_grad_norms.append(grad_norm)
        
        if branch_grad_norms:
            gradient_norms.append(np.mean(branch_grad_norms))
            gradient_variances.append(np.var(branch_grad_norms))
    
    return {
        'mean_gradient_norm': np.mean(gradient_norms),
        'gradient_norm_variance': np.var(gradient_norms),
        'mean_gradient_variance': np.mean(gradient_variances),
        'gradient_balance': np.std(gradient_norms) / (np.mean(gradient_norms) + 1e-8)
    }


def compute_diversity_metrics(
    representations: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute diversity metrics for branch representations
    
    Args:
        representations: List of representation tensors from branches
        
    Returns:
        Dictionary of diversity metrics
    """
    # Stack representations
    reps = torch.stack(representations)  # (num_branches, batch_size, features)
    
    # Compute pairwise distances
    distances = []
    for i in range(len(representations)):
        for j in range(i+1, len(representations)):
            dist = F.pairwise_distance(
                representations[i].flatten(1),
                representations[j].flatten(1)
            ).mean()
            distances.append(dist.item())
    
    # Compute orthogonality
    orthogonality_scores = []
    for i in range(len(representations)):
        for j in range(i+1, len(representations)):
            rep_i = representations[i].flatten(1)
            rep_j = representations[j].flatten(1)
            
            # Compute dot product (closer to 0 means more orthogonal)
            dot_product = (rep_i * rep_j).sum(dim=1).abs().mean()
            orthogonality_scores.append(dot_product.item())
    
    return {
        'mean_distance': np.mean(distances),
        'distance_variance': np.var(distances),
        'mean_orthogonality': np.mean(orthogonality_scores),
        'diversity_index': np.mean(distances) / (np.std(distances) + 1e-8)
    }