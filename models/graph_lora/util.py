"""
Utility functions for Graph LoRA (Low-Rank Adaptation) implementation.

This module provides essential utility functions for:
- Dataset loading and preprocessing
- Graph operations and transformations
- Loss functions (SMMD, contrastive learning)
- Training utilities and parameter management
- File operations and experiment configuration

Author: Graph LoRA Team
"""

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_dense_adj
import numpy as np
import yaml
from yaml import SafeLoader
from typing import Callable, Optional, Tuple, Dict, Any, Union


def mkdir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
        
    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")


def get_activation_function(act_type: str = 'leakyrelu') -> Callable:
    """
    Get activation function based on string identifier.
    
    Args:
        act_type (str): Type of activation function
            Options: 'leakyrelu', 'tanh', 'relu', 'prelu', 'sigmoid'
            
    Returns:
        Callable: Activation function
        
    Raises:
        ValueError: If activation type is not supported
    """
    activation_map = {
        'leakyrelu': F.leaky_relu,
        'tanh': torch.tanh,
        'relu': F.relu,
        'prelu': nn.PReLU(),
        'sigmoid': F.sigmoid  # Fixed typo from 'sigmiod'
    }
    
    if act_type not in activation_map:
        raise ValueError(f"Unsupported activation type: {act_type}. "
                        f"Available options: {list(activation_map.keys())}")
    
    return activation_map[act_type]


# Backward compatibility
def act(act_type: str = 'leakyrelu') -> Callable:
    """Backward compatibility wrapper for get_activation_function."""
    return get_activation_function(act_type)


def get_dataset(path: str, name: str):
    """
    Load and preprocess graph datasets.
    
    Supports both Amazon (Computer/Photo) and Planetoid (Cora/CiteSeer/PubMed) datasets
    with automatic feature normalization.
    
    Args:
        path (str): Root directory path for dataset storage
        name (str): Dataset name - must be one of:
                   ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']
    
    Returns:
        torch_geometric.data.Dataset: Loaded dataset with normalized features
        
    Raises:
        AssertionError: If dataset name is not supported
    """
    supported_datasets = ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']
    assert name in supported_datasets, f"Dataset {name} not supported. Choose from: {supported_datasets}"
    
    # Amazon datasets (Computers, Photo)
    if name in ['Computers', 'Photo']:
        return Amazon(path, name, T.NormalizeFeatures())
    # Planetoid datasets (Cora, CiteSeer, PubMed)
    else:
        return Planetoid(path, name, transform=T.NormalizeFeatures())


def initialize_weights(module: nn.Module) -> None:
    """
    Initialize weights for linear layers using Kaiming uniform initialization.
    
    This function is designed to be used with nn.Module.apply() for 
    initializing all linear layers in a neural network.
    
    Args:
        module (nn.Module): Neural network module to initialize
        
    Returns:
        None
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SMMDLoss(nn.Module):
    """
    Squared Maximum Mean Discrepancy (SMMD) Loss for domain adaptation.
    
    This loss function measures the discrepancy between two distributions
    using either linear or RBF (Gaussian) kernels. It's particularly useful
    for aligning feature distributions between source and target domains.
    
    Args:
        kernel_type (str): Type of kernel - 'linear' or 'rbf'
        kernel_mul (float): Multiplier for kernel bandwidth in RBF kernel
        kernel_num (int): Number of kernels to use in RBF kernel
        fix_sigma (float, optional): Fixed bandwidth for RBF kernel
    """
    
    def __init__(self, 
                 kernel_type: str = 'rbf', 
                 kernel_mul: float = 2.0, 
                 kernel_num: int = 5, 
                 fix_sigma: Optional[float] = None, 
                 **kwargs):
        super(SMMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        
        if kernel_type not in ['linear', 'rbf']:
            raise ValueError("kernel_type must be either 'linear' or 'rbf'")

    def gaussian_kernel(self, 
                       source: torch.Tensor, 
                       target: torch.Tensor, 
                       kernel_mul: float, 
                       kernel_num: int, 
                       fix_sigma: Optional[float]) -> torch.Tensor:
        """
        Compute Gaussian (RBF) kernel matrix between source and target samples.
        
        Args:
            source (torch.Tensor): Source domain samples [n_source, feature_dim]
            target (torch.Tensor): Target domain samples [n_target, feature_dim]
            kernel_mul (float): Kernel bandwidth multiplier
            kernel_num (int): Number of different bandwidths to use
            fix_sigma (float, optional): Fixed bandwidth value
            
        Returns:
            torch.Tensor: Kernel matrix [n_total, n_total] where n_total = n_source + n_target
        """
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        
        # Compute pairwise distances efficiently
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        # Calculate bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        # Multi-scale kernel computation
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) 
                     for bandwidth_temp in bandwidth_list]
        
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X: torch.Tensor, f_of_Y: torch.Tensor) -> torch.Tensor:
        """
        Compute linear MMD^2 distance between two distributions.
        
        Args:
            f_of_X (torch.Tensor): Samples from first distribution
            f_of_Y (torch.Tensor): Samples from second distribution
            
        Returns:
            torch.Tensor: Linear MMD^2 distance (scalar)
        """
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        return delta.dot(delta.T)

    def forward(self, 
                source: torch.Tensor, 
                target: torch.Tensor, 
                ppr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute SMMD loss between source and target distributions.
        
        Args:
            source (torch.Tensor): Source domain samples
            target (torch.Tensor): Target domain samples  
            ppr (torch.Tensor, optional): Personalized PageRank weights for weighted MMD
            
        Returns:
            torch.Tensor: SMMD loss value
        """
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = source.size(0)
            kernels = self.gaussian_kernel(
                source, target, 
                kernel_mul=self.kernel_mul, 
                kernel_num=self.kernel_num, 
                fix_sigma=self.fix_sigma
            )
            
            # Compute MMD terms
            if ppr is None:
                XX = torch.mean(kernels[:batch_size, :batch_size])
            else:
                XX = torch.mean(kernels[:batch_size, :batch_size] * ppr)
                
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            
            return torch.mean(XX + YY - XY - YX)


def get_ppr_matrix(dataset, alpha: float = 0.05) -> torch.Tensor:
    """
    Compute Personalized PageRank (PPR) matrix for a graph dataset.
    
    PPR provides a measure of node importance and proximity in the graph,
    useful for weighting node relationships in various graph learning tasks.
    
    Args:
        dataset: Graph dataset with edge_index attribute
        alpha (float): Teleport probability (typically 0.05-0.15)
        
    Returns:
        torch.Tensor: PPR matrix [num_nodes, num_nodes]
    """
    # Convert edge index to dense adjacency matrix
    A_tilde = to_dense_adj(dataset.edge_index)[0]
    num_nodes = A_tilde.shape[0]
    
    # Compute normalized adjacency matrix (symmetric normalization)
    degree = A_tilde.sum(dim=1)
    # Add small epsilon to avoid division by zero
    degree = torch.clamp(degree, min=1e-12)
    D_tilde = torch.diag(1 / torch.sqrt(degree))
    H = D_tilde @ A_tilde @ D_tilde
    
    # Compute PPR matrix: α(I - (1-α)H)^(-1)
    identity = torch.eye(num_nodes, device=A_tilde.device)
    ppr_matrix = alpha * torch.linalg.inv(identity - (1 - alpha) * H)
    
    return ppr_matrix


def get_ppr_weight(test_dataset) -> torch.Tensor:
    """
    Compute PPR-based weights for graph nodes.
    
    These weights can be used to emphasize important nodes or relationships
    in loss functions or attention mechanisms.
    
    Args:
        test_dataset: Graph dataset to compute PPR weights for
        
    Returns:
        torch.Tensor: Normalized PPR weights [num_nodes, num_nodes]
    """
    ppr_matrix = get_ppr_matrix(test_dataset)
    
    # Handle zero values by replacing with minimum non-zero value
    min_nonzero = ppr_matrix[ppr_matrix != 0].min()
    ppr_matrix = torch.where(ppr_matrix == 0, min_nonzero, ppr_matrix)
    
    # Apply logarithmic transformation
    ppr_matrix = torch.log(1 + 1 / ppr_matrix)
    
    # Row-wise normalization
    row_sums = ppr_matrix.sum(1, keepdim=True)
    ppr_weight = (ppr_matrix / row_sums) * ppr_matrix.shape[0]
    
    return ppr_weight


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print detailed information about model parameters.
    
    This function analyzes the model and prints:
    - Number of trainable parameters
    - Total number of parameters  
    - Percentage of trainable parameters
    
    Args:
        model (nn.Module): PyTorch model to analyze
        
    Returns:
        None
    """
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    trainable_percentage = 100 * trainable_params / all_params if all_params > 0 else 0
    
    print(f"Model Parameter Analysis:")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Total params: {all_params:,}")
    print(f"  Trainable percentage: {trainable_percentage:.2f}%")


def compute_similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        z1 (torch.Tensor): First set of embeddings [N, D]
        z2 (torch.Tensor): Second set of embeddings [M, D]
        
    Returns:
        torch.Tensor: Similarity matrix [N, M]
    """
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)
    return torch.mm(z1_norm, z2_norm.t())


# Backward compatibility
def sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Backward compatibility wrapper for compute_similarity."""
    return compute_similarity(z1, z2)


def batched_gct_loss(z1: torch.Tensor, 
                    z2: torch.Tensor, 
                    batch_size: int, 
                    mask: torch.Tensor, 
                    tau: float = 0.5) -> torch.Tensor:
    """
    Compute batched Graph Contrastive (GCT) loss for large graphs.
    
    This function processes the contrastive loss in batches to handle
    memory constraints when working with large graphs.
    
    Args:
        z1 (torch.Tensor): First view embeddings [N, D]
        z2 (torch.Tensor): Second view embeddings [N, D]  
        batch_size (int): Batch size for processing
        mask (torch.Tensor): Mask for positive pairs [N, N]
        tau (float): Temperature parameter for softmax
        
    Returns:
        torch.Tensor: Contrastive loss for each node [N]
    """
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / tau)
    indices = torch.arange(0, num_nodes, device=device)
    losses = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_nodes)
        idx = indices[start_idx:end_idx]
        
        # Compute similarity matrices for current batch
        refl_sim = f(compute_similarity(z1[idx], z1))  # [B, N]
        between_sim = f(compute_similarity(z1[idx], z2))  # [B, N]

        # Extract current batch mask
        batch_mask = mask[start_idx:end_idx]
        
        # Compute contrastive loss
        numerator = (batch_mask * between_sim).sum(1)
        denominator = (refl_sim.sum(1) + between_sim.sum(1) - 
                      refl_sim[:, start_idx:end_idx].diag())
        
        batch_loss = -torch.log(numerator / denominator)
        losses.append(batch_loss)

    return torch.cat(losses)


def batched_smmd_loss(z1: torch.Tensor, 
                     z2, 
                     MMD: SMMDLoss, 
                     ppr_weight: torch.Tensor, 
                     batch_size: int) -> torch.Tensor:
    """
    Compute batched SMMD loss for memory-efficient processing.
    
    Args:
        z1 (torch.Tensor): Source embeddings [N, D]
        z2: Target embeddings (iterator/loader)
        MMD (SMMDLoss): SMMD loss function
        ppr_weight (torch.Tensor): PPR weights [N, N]
        batch_size (int): Batch size for processing
        
    Returns:
        torch.Tensor: Average SMMD loss across batches
    """
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes, device=device)
    losses = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_nodes)
        mask = indices[start_idx:end_idx]
        
        # Extract PPR weights for current batch
        ppr = ppr_weight[mask][:, mask]
        
        # Get target batch (assuming z2 is an iterator)
        target = next(iter(z2))
        
        # Compute SMMD loss for current batch
        batch_loss = MMD(z1[mask], target, ppr)
        losses.append(batch_loss)

    return torch.stack(losses).mean()


def get_few_shot_mask(data, 
                     shot: int, 
                     dataname: str, 
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train/validation/test masks for few-shot learning scenarios.
    
    This function creates balanced splits where each class has exactly 'shot' 
    training examples, ensuring fair evaluation across all classes.
    
    Args:
        data: Graph data object with node labels and potentially existing masks
        shot (int): Number of training examples per class
        dataname (str): Dataset name to determine split strategy
        device (torch.device): Device to place tensors on
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            (train_mask, val_mask, test_mask) boolean tensors
    """
    np.random.seed(0)  # Ensure reproducible splits
    num_classes = int(data.y.max()) + 1
    y = data.y.cpu()
    num_nodes = len(y)
    
    selected_train_indices = []
    
    # Planetoid datasets have predefined splits
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        train_mask = data.train_mask.cpu()
        val_mask = data.val_mask.cpu()
        test_mask = data.test_mask.cpu()
        
        # Sample 'shot' examples per class from existing training set
        for class_idx in range(num_classes):
            class_train_indices = torch.arange(num_nodes)[(y == class_idx) & train_mask]
            if len(class_train_indices) >= shot:
                selected = np.random.choice(class_train_indices.numpy(), shot, replace=False)
                selected_train_indices.extend(selected)
        
        # Create new training mask with selected indices
        new_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        new_train_mask[selected_train_indices] = True
        
        return new_train_mask.to(device), val_mask.to(device), test_mask.to(device)
    
    # Amazon datasets need custom splits
    else:
        # Sample 'shot' examples per class from entire dataset
        for class_idx in range(num_classes):
            class_indices = torch.arange(num_nodes)[y == class_idx]
            if len(class_indices) >= shot:
                selected = np.random.choice(class_indices.numpy(), shot, replace=False)
                selected_train_indices.extend(selected)
        
        # Initialize all masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Set training indices
        train_mask[selected_train_indices] = True
        
        # Split remaining indices for validation and test
        remaining_indices = np.arange(num_nodes)[~train_mask.numpy()]
        np.random.shuffle(remaining_indices)
        
        # 20% of remaining for validation, 80% for test
        val_size = int(len(remaining_indices) * 0.2)
        val_mask[remaining_indices[:val_size]] = True
        test_mask[remaining_indices[val_size:]] = True
        
        return train_mask.to(device), val_mask.to(device), test_mask.to(device)


def get_parameter(args) -> Any:
    """
    Load hyperparameters from YAML configuration file.
    
    This function reads experiment configurations and updates the args object
    with dataset-specific and setting-specific hyperparameters.
    
    Args:
        args: Argument namespace with experiment configuration
        
    Returns:
        Updated args object with loaded hyperparameters
    """
    with open(args.para_config, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    # Determine experimental setting
    if args.few:
        setting = '10shot' if args.shot == 10 else '5shot'
    else:
        setting = 'public'
    
    # Extract dataset-specific parameters
    dataset_config = config[setting][args.test_dataset]
    
    # Update args with loaded parameters
    hyperparams = ['wd1', 'wd2', 'wd3', 'lr1', 'lr2', 'lr3', 
                   'l1', 'l2', 'l3', 'l4', 'num_epochs']
    
    for param in hyperparams:
        if param in dataset_config:
            if param == 'num_epochs':
                setattr(args, param, dataset_config[param])
            else:
                setattr(args, param, float(dataset_config[param]))
    
    return args
