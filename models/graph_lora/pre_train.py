"""
Pre-training module for Graph Neural Networks using self-supervised learning.

This module implements pre-training functionality for GNNs using contrastive learning
methods like GRACE (Graph Random neural network with Adaptive Contrast Enhancement).
The pre-trained models can be used as foundation models for downstream tasks.

Key Features:
- Self-supervised pre-training using graph contrastive learning
- Support for multiple GNN architectures (GCN, GAT, TransformerConv)
- Feature reduction capabilities for high-dimensional graphs
- Model checkpointing and resumption

Author: Graph LoRA Team
"""

import os
import torch
from time import time
from typing import Dict, Any, Optional
from torch_geometric.transforms import SVDFeatureReduction

# Import local modules
from .GNN_model import GNN
from .GRACE_model import GRACE
from .util import get_dataset, get_activation_function, mkdir


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate pre-training configuration parameters.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_params = [
        'output_dim', 'num_proj_dim', 'activation', 'learning_rate',
        'weight_decay', 'num_epochs', 'tau', 'gnn_type', 'num_layers',
        'drop_edge_rate', 'drop_feature_rate'
    ]
    
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
    
    # Validate parameter ranges
    if config['output_dim'] <= 0:
        raise ValueError("output_dim must be positive")
    if config['num_proj_dim'] <= 0:
        raise ValueError("num_proj_dim must be positive")
    if not 0 <= config['drop_edge_rate'] <= 1:
        raise ValueError("drop_edge_rate must be between 0 and 1")
    if not 0 <= config['drop_feature_rate'] <= 1:
        raise ValueError("drop_feature_rate must be between 0 and 1")
    if config['tau'] <= 0:
        raise ValueError("tau (temperature) must be positive")


def setup_device(gpu: int) -> torch.device:
    """
    Setup computing device (GPU/CPU) for training.
    
    Args:
        gpu (int): GPU device ID to use
        
    Returns:
        torch.device: Device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f"Using GPU device: {device}")
        print(f"GPU memory available: {torch.cuda.get_device_properties(gpu).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device


def load_and_preprocess_data(dataname: str, 
                           is_reduction: bool = False, 
                           reduction_dim: int = 100) -> tuple:
    """
    Load and preprocess graph dataset for pre-training.
    
    Args:
        dataname (str): Name of the dataset to load
        is_reduction (bool): Whether to apply dimensionality reduction
        reduction_dim (int): Target dimension for feature reduction
        
    Returns:
        tuple: (dataset, data, input_dim) - dataset object, graph data, and input dimension
    """
    print(f"Loading dataset: {dataname}")
    dataset_path = os.path.join('./datasets', dataname)
    
    try:
        dataset = get_dataset(dataset_path, dataname)
        data = dataset[0]
        
        print(f"Dataset loaded successfully:")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        print(f"  Original features: {data.x.shape[1]}")
        
        # Apply feature reduction if requested
        if is_reduction:
            print(f"Applying SVD feature reduction to {reduction_dim} dimensions")
            feature_reduce = SVDFeatureReduction(out_channels=reduction_dim)
            data = feature_reduce(data)
            print(f"  Reduced features: {data.x.shape[1]}")
        
        input_dim = data.x.shape[1]
        return dataset, data, input_dim
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataname}: {str(e)}")


def create_pretrain_model(input_dim: int, 
                         config: Dict[str, Any], 
                         pretext: str = 'GRACE') -> torch.nn.Module:
    """
    Create and initialize pre-training model.
    
    Args:
        input_dim (int): Input feature dimension
        config (Dict[str, Any]): Model configuration
        pretext (str): Pre-training method ('GRACE' or others)
        
    Returns:
        torch.nn.Module: Initialized pre-training model
    """
    print(f"Creating {pretext} pre-training model...")
    
    # Extract configuration parameters
    output_dim = config['output_dim']
    num_proj_dim = config['num_proj_dim']
    activation = get_activation_function(config['activation'])
    gnn_type = config['gnn_type']
    num_layers = config['num_layers']
    drop_edge_rate = config['drop_edge_rate']
    drop_feature_rate = config['drop_feature_rate']
    tau = config['tau']
    
    # Create base GNN encoder
    gnn = GNN(
        input_dim=input_dim,
        out_dim=output_dim,
        activation=activation,
        gnn_type=gnn_type,
        gnn_layer_num=num_layers
    )
    
    # Create pre-training wrapper
    if pretext == 'GRACE':
        pretrain_model = GRACE(
            gnn=gnn,
            num_hidden=output_dim,
            num_proj_hidden=num_proj_dim,
            drop_edge_rate=drop_edge_rate,
            drop_feature_rate=drop_feature_rate,
            tau=tau
        )
    else:
        # Default to GRACE for now, can be extended for other methods
        print(f"Warning: {pretext} not explicitly supported, using GRACE")
        pretrain_model = GRACE(
            gnn=gnn,
            num_hidden=output_dim,
            num_proj_hidden=num_proj_dim,
            drop_edge_rate=drop_edge_rate,
            drop_feature_rate=drop_feature_rate,
            tau=tau
        )
    
    print(f"Model created with {sum(p.numel() for p in pretrain_model.parameters()):,} parameters")
    return pretrain_model


def save_model_checkpoint(model: torch.nn.Module, 
                         model_path: str, 
                         dataname: str, 
                         pretext: str, 
                         gnn_type: str, 
                         current_loss: float, 
                         epoch: int) -> None:
    """
    Save model checkpoint with metadata.
    
    Args:
        model (torch.nn.Module): Model to save
        model_path (str): Path to save the model
        dataname (str): Dataset name
        pretext (str): Pre-training method
        gnn_type (str): GNN architecture type
        current_loss (float): Current training loss
        epoch (int): Current epoch number
    """
    try:
        # Save just the GNN encoder (not the full pre-training wrapper)
        torch.save(model.gnn.state_dict(), model_path)
        print(f"✓ Model saved: {model_path}")
        print(f"  Dataset: {dataname}, Method: {pretext}, GNN: {gnn_type}")
        print(f"  Epoch: {epoch}, Loss: {current_loss:.6f}")
    except Exception as e:
        print(f"✗ Failed to save model: {str(e)}")


def pretrain(dataname: str, 
            pretext: str, 
            config: Dict[str, Any], 
            gpu: int, 
            is_reduction: bool = False) -> None:
    """
    Main pre-training function for Graph Neural Networks.
    
    This function implements self-supervised pre-training using contrastive learning
    on graph-structured data. The pre-trained model can be used for downstream tasks.
    
    Args:
        dataname (str): Name of the dataset for pre-training
        pretext (str): Pre-training method (e.g., 'GRACE')
        config (Dict[str, Any]): Configuration dictionary containing:
            - output_dim: GNN output dimension
            - num_proj_dim: Projection head dimension  
            - activation: Activation function name
            - learning_rate: Learning rate for optimization
            - weight_decay: L2 regularization strength
            - num_epochs: Number of training epochs
            - tau: Temperature parameter for contrastive loss
            - gnn_type: GNN architecture (GCN, GAT, TransformerConv)
            - num_layers: Number of GNN layers
            - drop_edge_rate: Edge dropout rate for augmentation
            - drop_feature_rate: Feature dropout rate for augmentation
        gpu (int): GPU device ID to use
        is_reduction (bool): Whether to apply feature dimensionality reduction
        
    Returns:
        None
        
    Raises:
        ValueError: If configuration parameters are invalid
        RuntimeError: If pre-training fails
    """
    print("=" * 60)
    print("GRAPH NEURAL NETWORK PRE-TRAINING")
    print("=" * 60)
    print(f"Current working directory: {os.getcwd()}")
    
    # Validate configuration
    validate_config(config)
    
    # Setup device
    device = setup_device(gpu)
    
    # Load and preprocess data
    dataset, data, input_dim = load_and_preprocess_data(dataname, is_reduction)
    data = data.to(device)
    
    # Create output directory for pre-trained models
    pretrained_model_dir = './pre_trained_gnn/'
    mkdir(pretrained_model_dir)
    
    # Create pre-training model
    pretrain_model = create_pretrain_model(input_dim, config, pretext)
    pretrain_model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        pretrain_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    print(f"\nOptimizer configured:")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    
    # Training setup
    num_epochs = config['num_epochs']
    model_filename = f"{dataname}.{pretext}.{config['gnn_type']}.{is_reduction}.pth"
    model_path = os.path.join(pretrained_model_dir, model_filename)
    
    print(f"\nStarting pre-training for {num_epochs} epochs...")
    print(f"Model will be saved to: {model_path}")
    
    # Training loop
    start_time = time()
    prev_time = start_time
    min_loss = float('inf')
    best_epoch = 0
    
    pretrain_model.train()
    
    try:
        for epoch in range(1, num_epochs + 1):
            # Forward pass and loss computation
            optimizer.zero_grad()
            loss = pretrain_model.compute_loss(data.x, data.edge_index)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Timing and logging
            current_time = time()
            epoch_time = current_time - prev_time
            total_time = current_time - start_time
            
            print(f'Epoch {epoch:3d}/{num_epochs} | '
                  f'Loss: {loss.item():.6f} | '
                  f'Time: {epoch_time:.2f}s | '
                  f'Total: {total_time:.2f}s')
            
            # Save best model
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_epoch = epoch
                save_model_checkpoint(
                    pretrain_model, model_path, dataname, 
                    pretext, config['gnn_type'], loss.item(), epoch
                )
            
            prev_time = current_time
            
            # Memory cleanup for CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        raise RuntimeError(f"Pre-training failed: {str(e)}")
    
    # Training summary
    total_training_time = time() - start_time
    print("\n" + "=" * 60)
    print("PRE-TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best model saved at epoch {best_epoch}")
    print(f"Best loss: {min_loss:.6f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/num_epochs:.2f} seconds")
    print(f"Final model path: {model_path}")
    
    if torch.cuda.is_available():
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")