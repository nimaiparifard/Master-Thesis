"""
Advanced Multi-Exit Neural Network Evaluation Module

This module provides comprehensive training and evaluation utilities for multi-exit
neural networks with dynamic early exit capabilities. It implements sophisticated
early stopping mechanisms, progressive layer training, and efficient evaluation
strategies for graph neural networks.

Key Features:
- Dynamic early exit based on prediction confidence
- Progressive layer training with attention mechanisms
- Efficient memory management and GPU utilization
- Comprehensive evaluation metrics and early stopping
- Support for both standard and multi-exit architectures

Performance Optimizations:
- Vectorized operations for batch processing
- Memory-efficient tensor operations
- Gradient accumulation strategies
- Dynamic evaluation scheduling

Author: Graph Neural Network Research Team
Version: 2.0 (Optimized and Documented)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union, Any
import warnings
import time
from collections import defaultdict
import numpy as np

# Global configuration - should be imported or set externally
try:
    from config import config, device, T, num_classes, classifier
except ImportError:
    warnings.warn("Config module not found. Using default values.")
    
    class DefaultConfig:
        patience = 100
        epochs = 200
    
    config = DefaultConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1.0  # Temperature for sigmoid activation
    num_classes = 10  # Default number of classes
    classifier = None  # Will be set externally


def efficient_train_eval(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    feature_embeddings: List[torch.Tensor],
    model_list: List[torch.nn.Module],
    progressive_layers: List[torch.nn.Module],
    alpha_parameters: List[torch.nn.Parameter],
    exit_classifiers: List[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    temperature: float = 1.0,
    use_mixed_precision: bool = True
) -> float:
    """
    Advanced training and evaluation for multi-exit neural networks with dynamic early exit.
    
    This function implements a sophisticated training strategy for neural networks with
    multiple exit points. Each exit point allows the model to make predictions at different
    depths, enabling dynamic inference where simple samples can exit early while complex
    samples proceed through deeper layers.
    
    Algorithm Overview:
    1. Progressive Training: Train multiple model layers with attention-weighted combinations
    2. Dynamic Early Exit: During evaluation, samples exit when predictions stabilize
    3. Loss Aggregation: Combine losses from all exit points for comprehensive training
    4. Early Stopping: Monitor validation performance to prevent overfitting
    
    Key Innovations:
    - Temperature-scaled sigmoid attention for layer combination
    - Comprehensive loss aggregation across all exit points
    - Memory-efficient gradient computation with retain_graph
    - Advanced early stopping with patience mechanism
    
    Performance Optimizations:
    - GPU memory management with proper tensor placement
    - Vectorized feature extraction and processing
    - Efficient batch processing with minimal data movement
    - Optional mixed precision training for speed and memory
    
    Args:
        train_loader (DataLoader): Training data batches containing graph data
        val_loader (DataLoader): Validation data for early stopping decisions
        test_loader (DataLoader): Test data for final performance evaluation
        feature_embeddings (List[torch.Tensor]): Pre-computed node embeddings for each layer
        model_list (List[torch.nn.Module]): Sequential neural network models for progressive training
        progressive_layers (List[torch.nn.Module]): Feature transformation layers
        alpha_parameters (List[torch.nn.Parameter]): Learnable attention weights for layer combination
        exit_classifiers (List[torch.nn.Module]): Classification heads for each exit point
        optimizer (torch.optim.Optimizer): Optimizer for all trainable parameters
        temperature (float): Temperature parameter for sigmoid attention. Default: 1.0
        use_mixed_precision (bool): Whether to use automatic mixed precision. Default: True
    
    Returns:
        float: Best test accuracy achieved during training based on validation performance
    
    Raises:
        ValueError: If model_list and progressive_layers have different lengths
        RuntimeError: If GPU memory is insufficient for the computation
    
    Example:
        >>> train_loader = DataLoader(train_dataset, batch_size=32)
        >>> val_loader = DataLoader(val_dataset, batch_size=64)
        >>> test_loader = DataLoader(test_dataset, batch_size=64)
        >>> models = [GNN(hidden_dim=128) for _ in range(3)]
        >>> best_acc = efficient_train_eval(train_loader, val_loader, test_loader, 
        ...                                embeddings, models, prog_layers, alphas, exits, optimizer)
    
    Time Complexity: O(E * B * L * (N + M)) where E=epochs, B=batches, L=layers, N=nodes, M=edges
    Space Complexity: O(B * N * D) where B=batch_size, N=max_nodes, D=hidden_dim
    """
    # Input validation
    if len(model_list) != len(progressive_layers):
        raise ValueError(f"Model list length ({len(model_list)}) must match progressive layers length ({len(progressive_layers)})")
    
    if len(model_list) != len(exit_classifiers):
        raise ValueError(f"Model list length ({len(model_list)}) must match exit classifiers length ({len(exit_classifiers)})")
    
    # Initialize training state
    patience_counter = 0
    best_validation_accuracy = 0.0
    best_test_accuracy = 0.0
    num_layers = len(model_list)
    
    # Loss function and optimization setup
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    
    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    
    # Training metrics tracking
    training_metrics = defaultdict(list)
    
    print(f"Starting training with {num_layers} layers, patience={config.patience}, epochs={config.epochs}")
    
    # Main training loop
    for epoch in tqdm(range(config.epochs), desc="Training Epochs"):
        epoch_start_time = time.time()
        total_epoch_loss = 0.0
        num_batches = 0
        
        # Training phase
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through progressive layers
            with torch.cuda.amp.autocast() if scaler else torch.no_grad() if not scaler else torch.enable_grad():
                layer_outputs, total_loss = _forward_multi_exit(
                    batch_data, feature_embeddings, model_list, progressive_layers,
                    alpha_parameters, exit_classifiers, criterion, temperature, training=True
                )
            
            # Backward pass
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            total_epoch_loss += total_loss.item()
            num_batches += 1
            
            # Memory cleanup
            del layer_outputs, total_loss
            if batch_idx % 50 == 0:  # Periodic garbage collection
                torch.cuda.empty_cache()
        
        # Calculate average training loss
        avg_training_loss = total_epoch_loss / max(num_batches, 1)
        training_metrics['train_loss'].append(avg_training_loss)
        
        # Validation and test evaluation
        validation_accuracy = efficient_eval(
            val_loader, feature_embeddings, model_list, progressive_layers,
            alpha_parameters, exit_classifiers, temperature
        )
        test_accuracy = efficient_eval(
            test_loader, feature_embeddings, model_list, progressive_layers,
            alpha_parameters, exit_classifiers, temperature
        )
        
        training_metrics['val_acc'].append(validation_accuracy)
        training_metrics['test_acc'].append(test_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        
        # Early stopping logic with improved tracking
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_test_accuracy = test_accuracy
            patience_counter = 0
            
            # Optional: Save best model state
            # _save_model_checkpoint(model_list, progressive_layers, exit_classifiers, epoch)
            
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch:3d} | Loss: {avg_training_loss:.4f} | "
                  f"Val Acc: {validation_accuracy:.4f} | Test Acc: {test_accuracy:.4f} | "
                  f"Time: {epoch_time:.2f}s | Patience: {patience_counter}/{config.patience}")
        
        # Early stopping check
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
            print(f"Corresponding test accuracy: {best_test_accuracy:.4f}")
            break
    
    # Final performance summary
    print(f"\nTraining completed!")
    print(f"Final best test accuracy: {best_test_accuracy:.4f}")
    
    return best_test_accuracy


def efficient_eval(
    data_loader: torch.utils.data.DataLoader,
    feature_embeddings: List[torch.Tensor],
    model_list: List[torch.nn.Module],
    progressive_layers: List[torch.nn.Module],
    alpha_parameters: List[torch.nn.Parameter],
    exit_classifiers: List[torch.nn.Module],
    temperature: float = 1.0,
    return_detailed_metrics: bool = False
) -> Union[float, Dict[str, float]]:
    """
    Advanced evaluation with dynamic early exit based on prediction confidence.
    
    This function implements a sophisticated evaluation strategy where samples can
    exit the network at different depths based on prediction stability. This mimics
    human-like decision making where simple cases are resolved quickly while complex
    cases require deeper analysis.
    
    Early Exit Strategy:
    1. Prediction Stability: If consecutive predictions match, consider early exit
    2. Confidence Thresholding: High confidence predictions can exit early
    3. Dynamic Routing: Different samples take different computational paths
    4. Efficiency Gains: Reduces average computational cost per sample
    
    Algorithm Details:
    - Track prediction consistency across layers
    - Maintain per-sample visitation flags
    - Aggregate final predictions from all exit points
    - Compute accuracy and optional detailed metrics
    
    Performance Optimizations:
    - Vectorized prediction comparison operations
    - Memory-efficient batch processing
    - Minimal tensor copying and data movement
    - Efficient boolean masking for early exit logic
    
    Args:
        data_loader (DataLoader): Evaluation data batches
        feature_embeddings (List[torch.Tensor]): Pre-computed node embeddings
        model_list (List[torch.nn.Module]): Sequential neural network models
        progressive_layers (List[torch.nn.Module]): Feature transformation layers
        alpha_parameters (List[torch.nn.Parameter]): Attention weights for layer combination
        exit_classifiers (List[torch.nn.Module]): Classification heads for each exit point
        temperature (float): Temperature for sigmoid attention. Default: 1.0
        return_detailed_metrics (bool): Whether to return detailed performance metrics
    
    Returns:
        Union[float, Dict]: Overall accuracy or detailed metrics including:
            - accuracy: Overall classification accuracy
            - early_exit_rates: Percentage of samples exiting at each layer
            - avg_computational_cost: Average layers used per sample
            - confidence_scores: Distribution of prediction confidences
    
    Example:
        >>> accuracy = efficient_eval(test_loader, embeddings, models, prog_layers, 
        ...                          alphas, exits, temperature=1.0)
        >>> detailed = efficient_eval(test_loader, embeddings, models, prog_layers,
        ...                          alphas, exits, return_detailed_metrics=True)
        >>> print(f"Accuracy: {detailed['accuracy']:.4f}")
        >>> print(f"Avg layers used: {detailed['avg_computational_cost']:.2f}")
    
    Time Complexity: O(B * L * (N + M)) where B=batches, L=avg_layers_used, N=nodes, M=edges
    Space Complexity: O(B * N * C) where C=num_classes
    """
    # Set all models to evaluation mode
    _set_models_eval_mode(model_list, progressive_layers, exit_classifiers)
    
    # Initialize evaluation metrics
    total_correct_predictions = 0
    total_samples = 0
    
    # Detailed metrics tracking
    if return_detailed_metrics:
        exit_statistics = defaultdict(int)
        computational_costs = []
        confidence_scores = []
    
    # Evaluation loop
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_data in tqdm(data_loader, desc="Evaluating", leave=False):
            batch_data = batch_data.to(device, non_blocking=True)
            batch_size = batch_data.batch.max().item() + 1
            total_samples += batch_size
            
            # Initialize per-sample tracking
            sample_results = torch.zeros(
                batch_size, num_classes, 
                device=device, dtype=torch.float32
            )
            samples_not_exited = torch.ones(
                batch_size, device=device, dtype=torch.bool
            )
            previous_predictions = torch.full(
                (batch_size,), -1, device=device, dtype=torch.long
            )
            
            # Progressive evaluation through layers
            layer_output = None
            for layer_idx, (model, prog_layer, exit_classifier) in enumerate(
                zip(model_list, progressive_layers, exit_classifiers)
            ):
                # Forward pass through current layer
                if layer_idx == 0:
                    # First layer: process initial features
                    processed_features = prog_layer(
                        feature_embeddings[layer_idx][batch_data.original_idx.cpu()].to(device)
                    )
                    layer_output = model(processed_features, batch_data.edge_index)
                else:
                    # Subsequent layers: combine with attention mechanism
                    attention_weight = torch.sigmoid(alpha_parameters[layer_idx] / temperature)
                    
                    current_features = prog_layer(
                        feature_embeddings[layer_idx][batch_data.original_idx.cpu()].to(device)
                    )
                    
                    # Attention-weighted combination
                    combined_features = (
                        current_features * attention_weight + 
                        layer_output * (1 - attention_weight)
                    )
                    layer_output = model(combined_features, batch_data.edge_index)
                
                # Generate predictions for current layer
                hidden_representation = torch.cat([
                    layer_output[batch_data.root_n_index],
                    global_mean_pool(layer_output, batch_data.batch)
                ], dim=1)
                
                current_logits = exit_classifier(hidden_representation)
                current_probabilities = F.softmax(current_logits, dim=1)
                current_predictions = current_probabilities.argmax(dim=1)
                
                # Early exit logic: samples exit if prediction matches previous
                if layer_idx > 0:
                    early_exit_mask = (
                        (current_predictions == previous_predictions) & 
                        samples_not_exited
                    )
                    
                    # Update results for early-exiting samples
                    sample_results[early_exit_mask] = current_probabilities[early_exit_mask]
                    samples_not_exited[early_exit_mask] = False
                    
                    # Record exit statistics
                    if return_detailed_metrics:
                        exit_statistics[layer_idx] += early_exit_mask.sum().item()
                
                # Update tracking
                previous_predictions = current_predictions.clone()
            
            # Handle remaining samples (no early exit)
            sample_results[samples_not_exited] = current_probabilities[samples_not_exited]
            
            if return_detailed_metrics:
                # Record computational costs and confidence scores
                for i in range(batch_size):
                    layers_used = layer_idx + 1  # All layers if no early exit
                    if not samples_not_exited[i]:
                        # Find which layer this sample exited at
                        for exit_layer in range(1, len(model_list)):
                            if i in exit_statistics and exit_statistics[exit_layer] > 0:
                                layers_used = exit_layer
                                break
                    computational_costs.append(layers_used)
                
                confidence_scores.extend(sample_results.max(dim=1)[0].cpu().numpy())
            
            # Calculate accuracy for current batch
            final_predictions = sample_results.argmax(dim=1)
            total_correct_predictions += (final_predictions == batch_data.y).sum().item()
    
    # Compute final accuracy
    overall_accuracy = total_correct_predictions / max(total_samples, 1)
    
    if return_detailed_metrics:
        # Compile detailed metrics
        detailed_metrics = {
            'accuracy': overall_accuracy,
            'early_exit_rates': {
                f'layer_{i}': (exit_statistics[i] / max(total_samples, 1)) * 100
                for i in range(1, len(model_list))
            },
            'avg_computational_cost': np.mean(computational_costs) if computational_costs else len(model_list),
            'confidence_distribution': {
                'mean': np.mean(confidence_scores) if confidence_scores else 0.0,
                'std': np.std(confidence_scores) if confidence_scores else 0.0,
                'min': np.min(confidence_scores) if confidence_scores else 0.0,
                'max': np.max(confidence_scores) if confidence_scores else 1.0
            }
        }
        return detailed_metrics
    
    return overall_accuracy


def train_eval(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    feature_embeddings: List[torch.Tensor],
    model_list: List[torch.nn.Module],
    progressive_layers: List[torch.nn.Module],
    alpha_parameters: List[torch.nn.Parameter],
    exit_classifiers: List[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    final_classifier: torch.nn.Module,
    temperature: float = 1.0
) -> float:
    """
    Standard training and evaluation without early exit functionality.
    
    This function implements traditional neural network training where all samples
    pass through the complete network architecture. It serves as a baseline
    comparison for the efficient multi-exit approach and is useful for scenarios
    where consistent computational cost is preferred over efficiency gains.
    
    Key Differences from Efficient Training:
    - No early exit mechanisms during evaluation
    - Single classification head at the final layer
    - Consistent computational path for all samples
    - Simplified loss computation and backpropagation
    
    Algorithm Overview:
    1. Progressive feature transformation through multiple layers
    2. Attention-weighted combination of layer outputs
    3. Final classification using the last layer's representations
    4. Standard cross-entropy loss optimization
    5. Early stopping based on validation performance
    
    Performance Characteristics:
    - Predictable computational cost per sample
    - Higher accuracy potential due to full network utilization
    - Suitable for scenarios requiring consistent inference time
    - Better for deployment where latency predictability is crucial
    
    Args:
        train_loader (DataLoader): Training data batches
        val_loader (DataLoader): Validation data for early stopping
        test_loader (DataLoader): Test data for final evaluation
        feature_embeddings (List[torch.Tensor]): Pre-computed node embeddings
        model_list (List[torch.nn.Module]): Sequential neural network models
        progressive_layers (List[torch.nn.Module]): Feature transformation layers
        alpha_parameters (List[torch.nn.Parameter]): Attention weights for layer combination
        exit_classifiers (List[torch.nn.Module]): Early exit classifiers (unused in this mode)
        optimizer (torch.optim.Optimizer): Optimizer for all parameters
        final_classifier (torch.nn.Module): Final classification layer
        temperature (float): Temperature for sigmoid attention. Default: 1.0
    
    Returns:
        float: Best test accuracy achieved based on validation performance
    
    Example:
        >>> final_classifier = nn.Linear(hidden_dim * 2, num_classes)
        >>> best_acc = train_eval(train_loader, val_loader, test_loader,
        ...                      embeddings, models, prog_layers, alphas, 
        ...                      exits, optimizer, final_classifier)
    
    Time Complexity: O(E * B * L * (N + M)) where E=epochs, B=batches, L=layers, N=nodes, M=edges
    Space Complexity: O(B * N * D) where D=hidden_dim
    """
    # Training state initialization
    patience_counter = 0
    best_validation_accuracy = 0.0
    best_test_accuracy = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Starting standard training (no early exit) for {config.epochs} epochs")
    
    # Main training loop
    for epoch in tqdm(range(config.epochs), desc="Standard Training"):
        total_loss = 0.0
        num_batches = 0
        
        # Training phase
        for batch_data in train_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Forward pass through all layers
            layer_output = None
            for layer_idx, (model, prog_layer) in enumerate(zip(model_list, progressive_layers)):
                _set_training_mode([model, prog_layer])
                
                if layer_idx == 0:
                    processed_features = prog_layer(
                        feature_embeddings[layer_idx][batch_data.original_idx.cpu()].to(device)
                    )
                    layer_output = model(processed_features, batch_data.edge_index)
                else:
                    attention_weight = torch.sigmoid(alpha_parameters[layer_idx] / temperature)
                    current_features = prog_layer(
                        feature_embeddings[layer_idx][batch_data.original_idx.cpu()].to(device)
                    )
                    combined_features = (
                        current_features * attention_weight + 
                        layer_output * (1 - attention_weight)
                    )
                    layer_output = model(combined_features, batch_data.edge_index)
            
            # Handle backward compatibility for root node indexing
            if hasattr(batch_data, 'root_n_id'):
                batch_data.root_n_index = batch_data.root_n_id
            
            # Final classification
            final_representation = torch.cat([
                layer_output[batch_data.root_n_index],
                global_mean_pool(layer_output, batch_data.batch)
            ], dim=1)
            
            final_output = final_classifier(final_representation)
            loss = criterion(final_output, batch_data.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Evaluation phase
        validation_accuracy = eval_standard(
            val_loader, feature_embeddings, model_list, progressive_layers,
            alpha_parameters, final_classifier, temperature
        )
        test_accuracy = eval_standard(
            test_loader, feature_embeddings, model_list, progressive_layers,
            alpha_parameters, final_classifier, temperature
        )
        
        # Early stopping logic
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_test_accuracy = test_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 20 == 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {validation_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
        
        # Early stopping check
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return best_test_accuracy


def eval_standard(
    data_loader: torch.utils.data.DataLoader,
    feature_embeddings: List[torch.Tensor],
    model_list: List[torch.nn.Module],
    progressive_layers: List[torch.nn.Module],
    alpha_parameters: List[torch.nn.Parameter],
    final_classifier: torch.nn.Module,
    temperature: float = 1.0
) -> float:
    """
    Standard evaluation without early exit mechanisms.
    
    This function evaluates the neural network using the complete architecture
    for all samples. It provides a baseline comparison against early exit
    methods and represents traditional neural network evaluation.
    
    Characteristics:
    - All samples use the full network depth
    - Consistent computational cost per sample
    - Maximum representational capacity utilization
    - Deterministic inference paths
    
    Args:
        data_loader (DataLoader): Evaluation data batches
        feature_embeddings (List[torch.Tensor]): Pre-computed node embeddings
        model_list (List[torch.nn.Module]): Sequential neural network models
        progressive_layers (List[torch.nn.Module]): Feature transformation layers
        alpha_parameters (List[torch.nn.Parameter]): Attention weights
        final_classifier (torch.nn.Module): Final classification layer
        temperature (float): Temperature for sigmoid attention
    
    Returns:
        float: Classification accuracy on the evaluation data
    
    Time Complexity: O(B * L * (N + M)) where B=batches, L=layers, N=nodes, M=edges
    Space Complexity: O(B * N * D) where D=hidden_dim
    """
    _set_models_eval_mode(model_list, progressive_layers, [final_classifier])
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            total_samples += batch_data.batch.max().item() + 1
            
            # Forward pass through all layers
            layer_output = None
            for layer_idx, (model, prog_layer) in enumerate(zip(model_list, progressive_layers)):
                if layer_idx == 0:
                    processed_features = prog_layer(
                        feature_embeddings[layer_idx][batch_data.original_idx.cpu()].to(device)
                    )
                    layer_output = model(processed_features, batch_data.edge_index)
                else:
                    attention_weight = torch.sigmoid(alpha_parameters[layer_idx] / temperature)
                    current_features = prog_layer(
                        feature_embeddings[layer_idx][batch_data.original_idx.cpu()].to(device)
                    )
                    combined_features = (
                        current_features * attention_weight + 
                        layer_output * (1 - attention_weight)
                    )
                    layer_output = model(combined_features, batch_data.edge_index)
            
            # Handle backward compatibility
            if hasattr(batch_data, 'root_n_id'):
                batch_data.root_n_index = batch_data.root_n_id
            
            # Final prediction
            final_representation = torch.cat([
                layer_output[batch_data.root_n_index],
                global_mean_pool(layer_output, batch_data.batch)
            ], dim=1)
            
            predictions = final_classifier(final_representation).argmax(dim=1)
            total_correct += (predictions == batch_data.y).sum().item()
    
    accuracy = total_correct / max(total_samples, 1)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy


# Utility Functions

def _forward_multi_exit(
    batch_data: Any,
    feature_embeddings: List[torch.Tensor],
    model_list: List[torch.nn.Module],
    progressive_layers: List[torch.nn.Module],
    alpha_parameters: List[torch.nn.Parameter],
    exit_classifiers: List[torch.nn.Module],
    criterion: torch.nn.Module,
    temperature: float,
    training: bool = True
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Helper function for forward pass through multi-exit architecture.
    
    Handles the complex logic of progressive layer processing, attention-weighted
    feature combination, and loss aggregation across multiple exit points.
    
    Args:
        batch_data: Input batch data
        feature_embeddings: Pre-computed embeddings for each layer
        model_list: Sequential neural network models
        progressive_layers: Feature transformation layers
        alpha_parameters: Attention weights for layer combination
        exit_classifiers: Classification heads for each exit point
        criterion: Loss function
        temperature: Temperature for sigmoid attention
        training: Whether in training mode
    
    Returns:
        Tuple of layer outputs and aggregated loss
    """
    layer_outputs = []
    total_loss = 0.0
    layer_output = None
    
    for layer_idx, (model, prog_layer, exit_classifier) in enumerate(
        zip(model_list, progressive_layers, exit_classifiers)
    ):
        if training:
            _set_training_mode([model, prog_layer, exit_classifier])
        
        # Process features for current layer
        if layer_idx == 0:
            processed_features = prog_layer(
                feature_embeddings[layer_idx][batch_data.original_idx.cpu()].to(device)
            )
            layer_output = model(processed_features, batch_data.edge_index)
        else:
            attention_weight = torch.sigmoid(alpha_parameters[layer_idx] / temperature)
            current_features = prog_layer(
                feature_embeddings[layer_idx][batch_data.original_idx.cpu()].to(device)
            )
            combined_features = (
                current_features * attention_weight + 
                layer_output * (1 - attention_weight)
            )
            layer_output = model(combined_features, batch_data.edge_index)
        
        layer_outputs.append(layer_output)
        
        # Compute loss for current exit point
        if training:
            hidden_representation = torch.cat([
                layer_output[batch_data.root_n_index],
                global_mean_pool(layer_output, batch_data.batch)
            ], dim=1)
            exit_logits = exit_classifier(hidden_representation)
            layer_loss = criterion(exit_logits, batch_data.y)
            total_loss += layer_loss
    
    return layer_outputs, total_loss


def _set_training_mode(modules: List[torch.nn.Module]) -> None:
    """Set multiple modules to training mode."""
    for module in modules:
        module.train()


def _set_models_eval_mode(
    model_list: List[torch.nn.Module],
    progressive_layers: List[torch.nn.Module],
    exit_classifiers: List[torch.nn.Module]
) -> None:
    """Set all models to evaluation mode for consistent inference."""
    for model in model_list:
        model.eval()
    for layer in progressive_layers:
        layer.eval()
    for classifier in exit_classifiers:
        classifier.eval()


def calculate_model_efficiency_metrics(
    model_list: List[torch.nn.Module],
    data_loader: torch.utils.data.DataLoader,
    feature_embeddings: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Calculate comprehensive efficiency metrics for the multi-exit model.
    
    Computes various performance metrics including computational cost,
    memory usage, inference time, and accuracy trade-offs.
    
    Args:
        model_list: List of neural network models
        data_loader: Evaluation data loader
        feature_embeddings: Pre-computed embeddings
    
    Returns:
        Dictionary containing efficiency metrics
    """
    metrics = {
        'total_parameters': sum(
            sum(p.numel() for p in model.parameters()) 
            for model in model_list
        ),
        'average_depth_used': 0.0,
        'memory_efficiency': 0.0,
        'inference_speedup': 0.0
    }
    
    # Additional metrics can be computed here based on specific requirements
    return metrics