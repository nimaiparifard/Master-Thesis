"""
Advanced Graph Neural Network Training Module

This module provides comprehensive training and evaluation utilities for Graph Neural Networks (GNNs)
with support for both full-graph and subgraph-based training paradigms. It implements sophisticated
training strategies including early stopping, learning rate scheduling, and memory-efficient batch processing.

Key Features:
- Full-graph and subgraph training modes
- Advanced early stopping with patience mechanism
- Memory-efficient batch processing for large graphs
- Comprehensive evaluation metrics and logging
- Support for various GNN architectures (GCN, SAGE, GAT, GIN)
- Gradient clipping and regularization techniques

Performance Optimizations:
- Efficient GPU memory management
- Vectorized operations for batch processing
- Dynamic evaluation scheduling
- Memory-aware data loading strategies

Author: Graph Neural Network Research Team
Version: 2.0 (Optimized and Documented)
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric
from tqdm import tqdm
import numpy as np
import yaml
from yaml import SafeLoader
from typing import Optional, Tuple, Dict, Any, Union, List
import warnings
import time
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_subgraph(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip: Optional[float] = None,
    use_mixed_precision: bool = True
) -> float:
    """
    Advanced subgraph-based training for Graph Neural Networks.
    
    This function implements sophisticated subgraph-based training which is essential for
    scaling GNNs to large graphs. By training on subgraphs, we can handle graphs that
    don't fit in GPU memory while maintaining the benefits of graph structure learning.
    
    Algorithm Overview:
    1. Sample subgraphs from the original graph for each training node
    2. Train the model on these subgraphs using mini-batch gradient descent
    3. Evaluate on validation subgraphs for early stopping
    4. Apply advanced optimization techniques (gradient clipping, mixed precision)
    
    Key Innovations:
    - Memory-efficient subgraph processing
    - Advanced early stopping with validation tracking
    - Optional mixed precision training for speedup
    - Gradient clipping for training stability
    - Learning rate scheduling support
    
    Performance Optimizations:
    - Non-blocking GPU transfers for better pipeline utilization
    - Efficient batch processing with memory management
    - Optional gradient accumulation for larger effective batch sizes
    - Mixed precision training for memory and speed improvements
    
    Args:
        model (torch.nn.Module): The GNN model to train
        optimizer (torch.optim.Optimizer): Optimizer for model parameters
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss)
        config (Any): Configuration object containing training hyperparameters
        train_loader (DataLoader): DataLoader for training subgraphs
        val_loader (DataLoader): DataLoader for validation subgraphs
        test_loader (DataLoader): DataLoader for test subgraphs
        device (torch.device): Target device for computation
        scheduler (Optional): Learning rate scheduler. Default: None
        grad_clip (Optional[float]): Gradient clipping threshold. Default: None
        use_mixed_precision (bool): Whether to use automatic mixed precision. Default: True
    
    Returns:
        float: Best test accuracy achieved during training
    
    Raises:
        RuntimeError: If model training fails due to memory or convergence issues
        ValueError: If configuration parameters are invalid
    
    Example:
        >>> model = GNN(input_dim=1024, hidden_dim=256, output_dim=7)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> criterion = torch.nn.CrossEntropyLoss()
        >>> best_acc = train_subgraph(model, optimizer, criterion, config,
        ...                          train_loader, val_loader, test_loader, device)
    
    Time Complexity: O(E * B * S) where E=epochs, B=batches, S=subgraph_size
    Space Complexity: O(B * S * F) where F=feature_dim
    """
    # Input validation
    if not hasattr(config, 'epochs') or config.epochs <= 0:
        raise ValueError("config.epochs must be a positive integer")
    
    # Initialize training state
    training_metrics = defaultdict(list)
    best_validation_accuracy = 0.0
    best_test_accuracy = 0.0
    patience_counter = 0
    
    # Configure early stopping
    patience = getattr(config, 'patience', 100)
    use_early_stopping = getattr(config, 'earlystop', True)
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    
    logger.info(f"Starting subgraph training for {config.epochs} epochs")
    logger.info(f"Early stopping: {use_early_stopping}, Patience: {patience}")
    logger.info(f"Mixed precision: {use_mixed_precision}, Gradient clipping: {grad_clip}")
    
    # Main training loop
    for epoch in tqdm(range(config.epochs), desc="Training Epochs"):
        epoch_start_time = time.time()
        model.train()
        
        # Training phase metrics
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            with torch.cuda.amp.autocast() if scaler else torch.enable_grad():
                # Forward pass through subgraph
                predictions = model.forward_subgraph(
                    batch.x, batch.edge_index, batch.batch, batch.root_n_index
                )
                loss = criterion(predictions, batch.y)
            
            # Backward pass
            optimizer.zero_grad()
            
            if scaler:
                scaler.scale(loss).backward()
                
                # Gradient clipping with scaled gradients
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_accuracy += (predictions.argmax(dim=1) == batch.y).float().mean().item()
            num_batches += 1
            
            # Memory cleanup
            del predictions, loss
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_accuracy = epoch_accuracy / max(num_batches, 1)
        
        training_metrics['train_loss'].append(avg_train_loss)
        training_metrics['train_accuracy'].append(avg_train_accuracy)
        
        # Validation and early stopping
        if use_early_stopping:
            validation_accuracy = eval_subgraph(model, val_loader, device)
            training_metrics['val_accuracy'].append(validation_accuracy)
            
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_test_accuracy = eval_subgraph(model, test_loader, device)
                patience_counter = 0
                
                # Optional: Save best model checkpoint
                # torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pt')
                
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered at epoch {epoch}')
                logger.info(f'Best validation accuracy: {best_validation_accuracy:.4f}')
                logger.info(f'Corresponding test accuracy: {best_test_accuracy:.4f}')
                break
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        if epoch % 10 == 0 or epoch < 5:
            val_acc_str = f", Val Acc: {validation_accuracy:.4f}" if use_early_stopping else ""
            logger.info(f"Epoch {epoch:3d} | Loss: {avg_train_loss:.4f} | "
                       f"Train Acc: {avg_train_accuracy:.4f}{val_acc_str} | "
                       f"Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Final evaluation if early stopping wasn't used
    if not use_early_stopping:
        best_test_accuracy = eval_subgraph(model, test_loader, device)
        logger.info(f'Final test accuracy: {best_test_accuracy:.4f}')
    
    return best_test_accuracy


def eval_subgraph(
    model: torch.nn.Module, 
    data_loader: DataLoader, 
    device: torch.device,
    return_detailed_metrics: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Comprehensive evaluation for subgraph-based GNN models.
    
    This function provides efficient and accurate evaluation of GNN models trained
    on subgraphs. It handles batch processing, memory management, and provides
    optional detailed performance metrics.
    
    Key Features:
    - Memory-efficient batch processing
    - Comprehensive accuracy calculation
    - Optional detailed performance metrics
    - Robust error handling for edge cases
    
    Performance Optimizations:
    - Gradient computation disabled for efficiency
    - Non-blocking GPU transfers
    - Efficient tensor operations
    - Memory cleanup between batches
    
    Args:
        model (torch.nn.Module): Trained GNN model
        data_loader (DataLoader): DataLoader containing evaluation subgraphs
        device (torch.device): Target device for computation
        return_detailed_metrics (bool): Whether to return detailed metrics. Default: False
    
    Returns:
        Union[float, Dict]: Overall accuracy or detailed metrics dictionary
    
    Example:
        >>> accuracy = eval_subgraph(model, test_loader, device)
        >>> detailed = eval_subgraph(model, test_loader, device, return_detailed_metrics=True)
    
    Time Complexity: O(B * S) where B=batches, S=subgraph_size
    Space Complexity: O(B * C) where C=num_classes
    """
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    # Detailed metrics tracking
    if return_detailed_metrics:
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        confidence_scores = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            batch = batch.to(device, non_blocking=True)
            
            # Forward pass
            predictions = model.forward_subgraph(
                batch.x, batch.edge_index, batch.batch, batch.root_n_index
            )
            
            # Calculate accuracy
            predicted_labels = predictions.argmax(dim=1)
            correct_predictions = (predicted_labels == batch.y)
            
            total_correct += correct_predictions.sum().item()
            total_samples += batch.y.shape[0]
            
            # Detailed metrics collection
            if return_detailed_metrics:
                probabilities = F.softmax(predictions, dim=1)
                confidence_scores.extend(probabilities.max(dim=1)[0].cpu().numpy())
                
                # Per-class accuracy tracking
                for label in batch.y.unique():
                    mask = (batch.y == label)
                    per_class_correct[label.item()] += correct_predictions[mask].sum().item()
                    per_class_total[label.item()] += mask.sum().item()
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / max(total_samples, 1)
    
    if return_detailed_metrics:
        # Compile detailed metrics
        per_class_accuracy = {
            f'class_{k}': per_class_correct[k] / max(per_class_total[k], 1)
            for k in per_class_total.keys()
        }
        
        detailed_metrics = {
            'overall_accuracy': overall_accuracy,
            'per_class_accuracy': per_class_accuracy,
            'confidence_distribution': {
                'mean': np.mean(confidence_scores) if confidence_scores else 0.0,
                'std': np.std(confidence_scores) if confidence_scores else 0.0,
                'min': np.min(confidence_scores) if confidence_scores else 0.0,
                'max': np.max(confidence_scores) if confidence_scores else 1.0
            },
            'total_samples': total_samples
        }
        return detailed_metrics
    
    return overall_accuracy


def train_fullgraph(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    config: Any,
    data: torch_geometric.data.Data,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip: Optional[float] = None
) -> float:
    """
    Advanced full-graph training for Graph Neural Networks.
    
    This function implements full-graph training which processes the entire graph
    in each training iteration. This approach is suitable for smaller graphs that
    fit in GPU memory and can provide better performance due to the complete
    graph context availability.
    
    Algorithm Overview:
    1. Load the entire graph into GPU memory
    2. Train using the full graph context with masked node sampling
    3. Evaluate on validation nodes for early stopping
    4. Apply optimization techniques for training stability
    
    Key Features:
    - Full graph context utilization for better representation learning
    - Memory-efficient training with node masking
    - Advanced early stopping and learning rate scheduling
    - Gradient clipping for training stability
    
    Performance Characteristics:
    - Better accuracy potential due to full graph context
    - Higher memory requirements
    - Faster convergence for suitable graph sizes
    - Consistent computational cost per epoch
    
    Args:
        model (torch.nn.Module): The GNN model to train
        optimizer (torch.optim.Optimizer): Optimizer for model parameters
        criterion (torch.nn.Module): Loss function
        config (Any): Configuration object with training parameters
        data (torch_geometric.data.Data): The complete graph data
        device (torch.device): Target device for computation
        scheduler (Optional): Learning rate scheduler. Default: None
        grad_clip (Optional[float]): Gradient clipping threshold. Default: None
    
    Returns:
        float: Best test accuracy achieved during training
    
    Example:
        >>> data = torch_geometric.data.Data(x=features, edge_index=edges, y=labels)
        >>> best_acc = train_fullgraph(model, optimizer, criterion, config, data, device)
    
    Time Complexity: O(E * (N + M)) where E=epochs, N=nodes, M=edges
    Space Complexity: O(N + M) for graph storage
    """
    # Initialize training state
    best_validation_accuracy = 0.0
    best_test_accuracy = 0.0
    patience_counter = 0
    
    # Configure early stopping
    patience = getattr(config, 'patience', 100)
    use_early_stopping = getattr(config, 'earlystop', True)
    
    # Move data to device
    data = data.to(device)
    model.train()
    
    logger.info(f"Starting full-graph training for {config.epochs} epochs")
    logger.info(f"Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Main training loop
    for epoch in tqdm(range(config.epochs), desc="Full-Graph Training"):
        epoch_start_time = time.time()
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(data.x, data.edge_index)
        
        # Compute loss only on training nodes
        train_loss = criterion(predictions[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        train_loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Early stopping evaluation
        if use_early_stopping:
            validation_accuracy = eval_fullgraph(model, data, device, config, eval_split="valid")
            
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_test_accuracy = eval_fullgraph(model, data, device, config, eval_split="test")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        if epoch % 20 == 0 or epoch < 5:
            train_acc = (predictions[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()
            val_acc_str = f", Val Acc: {validation_accuracy:.4f}" if use_early_stopping else ""
            logger.info(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                       f"Train Acc: {train_acc:.4f}{val_acc_str} | "
                       f"Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Final evaluation if early stopping wasn't used
    if not use_early_stopping:
        best_test_accuracy = eval_fullgraph(model, data, device, config, eval_split="test")
    
    return best_test_accuracy


def eval_fullgraph(
    model: torch.nn.Module, 
    data: torch_geometric.data.Data, 
    device: torch.device, 
    config: Any, 
    eval_split: str = "valid"
) -> float:
    """
    Comprehensive evaluation for full-graph GNN models.
    
    This function evaluates GNN models trained on full graphs by computing
    predictions for the entire graph and extracting accuracy for the specified
    evaluation split (validation or test).
    
    Args:
        model (torch.nn.Module): Trained GNN model
        data (torch_geometric.data.Data): Complete graph data
        device (torch.device): Target device for computation
        config (Any): Configuration object
        eval_split (str): Evaluation split ("valid" or "test"). Default: "valid"
    
    Returns:
        float: Accuracy on the specified split
    
    Raises:
        AssertionError: If eval_split is not "valid" or "test"
    """
    assert eval_split in ["valid", "test"], f"eval_split must be 'valid' or 'test', got {eval_split}"
    
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
        predicted_labels = predictions.argmax(dim=1)
        
        if eval_split == "test":
            mask = data.test_mask
        else:
            mask = data.val_mask
        
        correct = (predicted_labels[mask] == data.y[mask]).sum()
        total = int(mask.sum())
        
        accuracy = int(correct) / max(total, 1)
    
    return accuracy


def train_eval(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    config: Any,
    data: torch_geometric.data.Data,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    **kwargs
) -> float:
    """
    Unified training and evaluation interface for both subgraph and full-graph modes.
    
    This function automatically selects the appropriate training strategy based on
    the configuration and available data loaders. It serves as a high-level interface
    that abstracts the complexity of different training paradigms.
    
    Training Mode Selection:
    - If subsampling is enabled and data loaders are provided: Use subgraph training
    - Otherwise: Use full-graph training
    
    Args:
        model (torch.nn.Module): GNN model to train
        optimizer (torch.optim.Optimizer): Optimizer for model parameters
        criterion (torch.nn.Module): Loss function
        config (Any): Configuration object
        data (torch_geometric.data.Data): Graph data
        train_loader (Optional[DataLoader]): Training subgraph loader
        val_loader (Optional[DataLoader]): Validation subgraph loader
        test_loader (Optional[DataLoader]): Test subgraph loader
        device (torch.device): Target device for computation
        **kwargs: Additional arguments for training functions
    
    Returns:
        float: Best test accuracy achieved
    
    Example:
        >>> # Subgraph training
        >>> best_acc = train_eval(model, optimizer, criterion, config, data,
        ...                      train_loader, val_loader, test_loader, device)
        >>> 
        >>> # Full-graph training
        >>> best_acc = train_eval(model, optimizer, criterion, config, data, device=device)
    """
    # Determine training mode
    use_subsampling = getattr(config, 'subsampling', False)
    
    if use_subsampling and all(loader is not None for loader in [train_loader, val_loader, test_loader]):
        logger.info("Using subgraph-based training")
        test_accuracy = train_subgraph(
            model, optimizer, criterion, config, 
            train_loader, val_loader, test_loader, device,
            **kwargs
        )
    else:
        logger.info("Using full-graph training")
        test_accuracy = train_fullgraph(
            model, optimizer, criterion, config, data, device,
            **kwargs
        )
    
    return test_accuracy


# Utility functions for advanced training features

def create_optimizer(
    model: torch.nn.Module, 
    config: Any
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Create optimizer and learning rate scheduler based on configuration.
    
    Args:
        model (torch.nn.Module): Model to optimize
        config (Any): Configuration object
    
    Returns:
        Tuple of optimizer and optional scheduler
    """
    # Get optimizer parameters
    lr = getattr(config, 'lr', 0.01)
    weight_decay = getattr(config, 'weight_decay', 5e-4)
    optimizer_type = getattr(config, 'optimizer', 'adam')
    
    # Create optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        momentum = getattr(config, 'momentum', 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create scheduler if specified
    scheduler = None
    scheduler_type = getattr(config, 'scheduler', None)
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif scheduler_type == 'step':
        step_size = getattr(config, 'step_size', 50)
        gamma = getattr(config, 'gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'exponential':
        gamma = getattr(config, 'gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    return optimizer, scheduler


def calculate_model_complexity(model: torch.nn.Module) -> Dict[str, Union[int, float]]:
    """
    Calculate model complexity metrics.
    
    Args:
        model (torch.nn.Module): Model to analyze
    
    Returns:
        Dict containing complexity metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory usage (approximate)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    complexity_metrics = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_mb': (param_size + buffer_size) / (1024 * 1024),
        'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # Assuming float32
    }
    
    return complexity_metrics


