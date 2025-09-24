"""
Training utilities for PGN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple, Callable
from tqdm import tqdm
import numpy as np

from ..utils.analysis import analyze_representations, compute_gradient_metrics


class Trainer:
    """
    Trainer class for PGN and baseline models
    
    Args:
        model: Model to train
        device: Device to use for training
        criterion: Loss function
        optimizer: Optimizer (if None, will use Adam)
        scheduler: Learning rate scheduler
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[Any] = None):
        
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self, 
                   train_loader: DataLoader,
                   epoch: int,
                   verbose: bool = True) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary of metrics for this epoch
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        loader = tqdm(train_loader, desc=f'Epoch {epoch}') if verbose else train_loader
        
        for batch_idx, (data, labels) in enumerate(loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Flatten if needed
            if len(data.shape) > 2:
                data = data.view(data.size(0), -1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if verbose and batch_idx % 10 == 0:
                loader.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self, 
                val_loader: DataLoader,
                verbose: bool = True) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            verbose: Whether to show progress
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        loader = tqdm(val_loader, desc='Validation') if verbose else val_loader
        
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Flatten if needed
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'accuracy': val_acc}
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 10,
             verbose: bool = True,
             analyze_freq: Optional[int] = None,
             early_stopping_patience: Optional[int] = None) -> Dict[str, list]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            verbose: Whether to show progress
            analyze_freq: Frequency to run analysis (for PGN)
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch + 1, verbose)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, verbose)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val Acc: {val_metrics['accuracy']:.2f}%")
                
                # Early stopping
                if early_stopping_patience:
                    if val_metrics['accuracy'] > best_val_acc:
                        best_val_acc = val_metrics['accuracy']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.2f}%")
            
            # Analysis for PGN
            if analyze_freq and (epoch + 1) % analyze_freq == 0:
                if hasattr(self.model, 'analyze_representations'):
                    sample_data, _ = next(iter(val_loader or train_loader))
                    sample_data = sample_data.to(self.device)
                    if len(sample_data.shape) > 2:
                        sample_data = sample_data.view(sample_data.size(0), -1)
                    
                    analysis = analyze_representations(self.model, sample_data)
                    
                    if verbose:
                        print(f"\nPGN Analysis at Epoch {epoch+1}:")
                        print(f"  Mean Geometric Distance: {analysis['mean_geometric_distance']:.4f}")
                        if analysis['mean_temporal_correlation'] is not None:
                            print(f"  Mean Temporal Correlation: {analysis['mean_temporal_correlation']:.4f}")
                        print(f"  Branch Specialization: {analysis['branch_specialization']:.4f}")
                        print(f"  Mean Effective Rank: {analysis['mean_effective_rank']:.2f}\n")
                    
                    # Store analysis in history
                    for key, value in analysis.items():
                        if key not in self.history:
                            self.history[key] = []
                        if isinstance(value, (int, float)):
                            self.history[key].append(value)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if val_loader is not None:
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
        
        return self.history
    
    def evaluate(self, 
                test_loader: DataLoader,
                return_predictions: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary of test metrics and optionally predictions
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_outputs = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Flatten if needed
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                # Forward pass
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                # Statistics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if return_predictions:
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_outputs.extend(outputs.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        results = {'accuracy': accuracy, 'total': total, 'correct': correct}
        
        if return_predictions:
            results['predictions'] = np.array(all_predictions)
            results['labels'] = np.array(all_labels)
            results['outputs'] = np.array(all_outputs)
        
        return results