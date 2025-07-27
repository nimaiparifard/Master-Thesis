"""
GraphLoRA: Graph Low-Rank Adaptation for Parameter-Efficient Transfer Learning

This module implements the GraphLoRA method for efficient transfer learning on graphs.
GraphLoRA combines graph contrastive learning with Low-Rank Adaptation (LoRA) to enable
parameter-efficient fine-tuning of pre-trained Graph Neural Networks.

Key Features:
- Parameter-efficient transfer learning using LoRA
- Graph contrastive learning for representation alignment
- Domain adaptation with SMMD loss
- Few-shot learning support
- Multi-objective optimization with reconstruction loss

Components:
- Feature projector for domain adaptation
- LoRA-enhanced GNN for efficient fine-tuning  
- Contrastive learning with InfoNCE loss
- Graph reconstruction loss for structure preservation
- Support for various datasets and evaluation protocols

Author: Graph LoRA Team
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops
from torch_geometric.loader import DataLoader
from typing import Tuple, Dict, Any, Optional

# Import local modules  
from .GNN_model import GNN, GNNLoRA
from .util import (get_dataset, get_activation_function, SMMDLoss, mkdir, 
                   get_ppr_weight, get_few_shot_mask, batched_smmd_loss, 
                   batched_gct_loss, print_trainable_parameters)


class Projector(nn.Module):
    """
    Feature projection module for domain adaptation.
    
    This module projects features from the target domain to match the
    source domain feature space, enabling effective transfer learning
    between different graph datasets.
    
    Args:
        input_size (int): Input feature dimension (target domain)
        output_size (int): Output feature dimension (source domain)
        
    Attributes:
        fc (nn.Linear): Linear projection layer
    """
    
    def __init__(self, input_size: int, output_size: int):
        super(Projector, self).__init__()
        
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Input and output sizes must be positive")
        
        self.fc = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input features to target dimension.
        
        Args:
            x (torch.Tensor): Input features [num_nodes, input_size]
            
        Returns:
            torch.Tensor: Projected features [num_nodes, output_size]
        """
        return self.fc(x)

    def initialize(self) -> None:
        """Initialize projection weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def __repr__(self) -> str:
        """String representation of the projector."""
        return f"Projector(in={self.fc.in_features}, out={self.fc.out_features})"


class LogReg(nn.Module):
    """
    Logistic regression classifier for downstream tasks.
    
    This simple linear classifier is used for node classification tasks
    after feature extraction by the GNN encoder.
    
    Args:
        hid_dim (int): Hidden dimension (input from GNN)
        out_dim (int): Output dimension (number of classes)
        
    Attributes:
        fc (nn.Linear): Linear classification layer
    """
    
    def __init__(self, hid_dim: int, out_dim: int):
        super(LogReg, self).__init__()
        
        if hid_dim <= 0 or out_dim <= 0:
            raise ValueError("Hidden and output dimensions must be positive")
        
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings [num_nodes, hid_dim]
            
        Returns:
            torch.Tensor: Classification logits [num_nodes, out_dim]
        """
        return self.fc(x)
    
    def initialize(self) -> None:
        """Initialize classification weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def __repr__(self) -> str:
        """String representation of the classifier."""
        return f"LogReg(in={self.fc.in_features}, out={self.fc.out_features})"


def setup_device(gpu_id: int) -> torch.device:
    """
    Setup computing device for training.
    
    Args:
        gpu_id (int): GPU device ID
        
    Returns:
        torch.device: Configured device
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {device}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU (CUDA not available)")
    
    return device


def load_datasets(args, is_reduction: bool = False) -> Tuple[Any, Any]:
    """
    Load and preprocess source and target datasets.
    
    Args:
        args: Arguments containing dataset names and configurations
        is_reduction (bool): Whether to apply dimensionality reduction
        
    Returns:
        Tuple containing (pretrain_dataset, test_dataset)
    """
    print(f"Loading datasets: {args.pretrain_dataset} -> {args.test_dataset}")
    
    # Load datasets
    pretrain_datapath = os.path.join('./datasets', args.pretrain_dataset)
    test_datapath = os.path.join('./datasets', args.test_dataset)
    
    pretrain_dataset = get_dataset(pretrain_datapath, args.pretrain_dataset)[0]
    test_dataset = get_dataset(test_datapath, args.test_dataset)[0]
    
    print(f"Source dataset: {pretrain_dataset.x.shape[0]} nodes, {pretrain_dataset.x.shape[1]} features")
    print(f"Target dataset: {test_dataset.x.shape[0]} nodes, {test_dataset.x.shape[1]} features")
    
    # Apply feature reduction if requested
    if is_reduction:
        print("Applying SVD feature reduction to 100 dimensions")
        feature_reduce = SVDFeatureReduction(out_channels=100)
        pretrain_dataset = feature_reduce(pretrain_dataset)
        test_dataset = feature_reduce(test_dataset)
        print(f"After reduction: source {pretrain_dataset.x.shape[1]}, target {test_dataset.x.shape[1]} features")
    
    # Add self-loops to edge indices
    pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]
    test_dataset.edge_index = add_remaining_self_loops(test_dataset.edge_index)[0]
    
    return pretrain_dataset, test_dataset


def setup_models(args, config: Dict[str, Any], pretrain_dataset, test_dataset, device: torch.device) -> Tuple:
    """
    Initialize and configure all model components.
    
    Args:
        args: Experiment arguments
        config: Model configuration dictionary
        pretrain_dataset: Source domain dataset
        test_dataset: Target domain dataset
        device: Computing device
        
    Returns:
        Tuple of (gnn2, projector, logreg, SMMD)
    """
    print("Setting up models...")
    
    # Create base GNN and load pre-trained weights
    gnn = GNN(
        pretrain_dataset.x.shape[1], 
        config['output_dim'], 
        get_activation_function(config['activation']), 
        config['gnn_type'], 
        config['num_layers']
    )
    
    # Load pre-trained weights
    model_path = f"./pre_trained_gnn/{args.pretrain_dataset}.{args.pretext}.{config['gnn_type']}.{args.is_reduction}.pth"
    if os.path.exists(model_path):
        gnn.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model: {model_path}")
    else:
        print(f"Warning: Pre-trained model not found at {model_path}")
    
    gnn.to(device)
    gnn.eval()
    
    # Freeze base GNN parameters
    for param in gnn.parameters():
        param.requires_grad = False
    
    # Create LoRA-enhanced GNN
    gnn2 = GNNLoRA(
        pretrain_dataset.x.shape[1], 
        config['output_dim'], 
        get_activation_function(config['activation']), 
        gnn, 
        config['gnn_type'], 
        config['num_layers'], 
        r=args.r
    )
    gnn2.to(device)
    gnn2.train()
    
    # Create feature projector
    projector = Projector(test_dataset.x.shape[1], pretrain_dataset.x.shape[1])
    projector.to(device)
    projector.train()
    
    # Create SMMD loss for domain adaptation
    SMMD = SMMDLoss().to(device)
    
    # Create classifier
    num_classes = int(test_dataset.y.max()) + 1
    logreg = LogReg(config['output_dim'], num_classes)
    logreg.to(device)
    
    # Print model information
    print_trainable_parameters(gnn2)
    print(f"Feature projector: {projector}")
    print(f"Classifier: {logreg}")
    
    return gnn2, projector, logreg, SMMD


def setup_data_splits(args, test_dataset, device: torch.device) -> Tuple:
    """
    Setup train/validation/test splits for the target dataset.
    
    Args:
        args: Experiment arguments
        test_dataset: Target dataset
        device: Computing device
        
    Returns:
        Tuple of (train_mask, val_mask, test_mask, updated_test_dataset)
    """
    print("Setting up data splits...")
    
    # Handle different dataset types and few-shot scenarios
    if args.test_dataset in ['PubMed', 'CiteSeer', 'Cora']:
        if args.few:
            train_mask, val_mask, test_mask = get_few_shot_mask(
                test_dataset, args.shot, args.test_dataset, device
            )
            print(f"Few-shot setup: {args.shot} samples per class")
        else:
            train_mask = test_dataset.train_mask.to(device)
            val_mask = test_dataset.val_mask.to(device)
            test_mask = test_dataset.test_mask.to(device)
            print("Using standard dataset splits")
    else:
        if args.few:
            train_mask, val_mask, test_mask = get_few_shot_mask(
                test_dataset, args.shot, args.test_dataset, device
            )
            print(f"Few-shot setup: {args.shot} samples per class")
        else:
            # Create random splits for Amazon datasets
            num_nodes = test_dataset.x.shape[0]
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            
            # 10% train, 10% val, 80% test
            train_size = int(num_nodes * 0.1)
            val_size = int(num_nodes * 0.1)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
            print("Created random splits (10% train, 10% val, 80% test)")
    
    # Update dataset masks
    test_dataset.train_mask = train_mask
    test_dataset.val_mask = val_mask
    test_dataset.test_mask = test_mask
    
    # Print split statistics
    print(f"Split sizes: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    
    return train_mask, val_mask, test_mask, test_dataset


def create_contrastive_mask(test_dataset, train_mask, args, device: torch.device) -> torch.Tensor:
    """
    Create mask for contrastive learning based on class labels.
    
    Args:
        test_dataset: Target dataset
        train_mask: Training node mask
        args: Experiment arguments
        device: Computing device
        
    Returns:
        torch.Tensor: Contrastive learning mask
    """
    print("Creating contrastive learning mask...")
    
    num_nodes = test_dataset.x.shape[0]
    num_classes = int(test_dataset.y.max()) + 1
    
    # Create mask where same-class nodes are considered positive pairs
    mask = torch.zeros((num_nodes, num_nodes), device=device)
    train_indices = torch.nonzero(train_mask, as_tuple=False).squeeze()
    train_labels = test_dataset.y[train_indices]
    
    # For each class, create positive pairs among training nodes
    idx_a = torch.tensor([], device=device, dtype=torch.long)
    idx_b = torch.tensor([], device=device, dtype=torch.long)
    
    for class_id in range(num_classes):
        class_nodes = train_indices[train_labels == class_id]
        if len(class_nodes) > 1:
            # Create all pairs within the class
            class_a = class_nodes.repeat_interleave(len(class_nodes))
            class_b = class_nodes.repeat(len(class_nodes))
            idx_a = torch.cat((idx_a, class_a))
            idx_b = torch.cat((idx_b, class_b))
    
    # Create sparse mask and convert to dense
    if len(idx_a) > 0:
        mask = torch.sparse_coo_tensor(
            indices=torch.stack((idx_a, idx_b)), 
            values=torch.ones(len(idx_a), device=device),
            size=[num_nodes, num_nodes]
        ).to_dense()
    
    # Apply supervision weight and add identity
    mask = args.sup_weight * (mask - torch.diag_embed(torch.diag(mask))) + torch.eye(num_nodes, device=device)
    
    print(f"Created contrastive mask with {len(idx_a)} positive pairs")
    return mask


def compute_reconstruction_loss(logits: torch.Tensor, 
                              target_adj: torch.Tensor, 
                              device: torch.device) -> torch.Tensor:
    """
    Compute graph reconstruction loss.
    
    Args:
        logits: Node classification logits
        target_adj: Target adjacency matrix
        device: Computing device
        
    Returns:
        torch.Tensor: Reconstruction loss
    """
    # Compute predicted adjacency from softmax similarities
    softmax_logits = torch.softmax(logits, dim=1)
    pred_adj = torch.sigmoid(torch.matmul(softmax_logits, softmax_logits.T))
    
    # Compute class-balanced weights
    pos_weight = float(target_adj.shape[0] ** 2 - target_adj.sum()) / target_adj.sum()
    weight_mask = target_adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), device=device)
    weight_tensor[weight_mask] = pos_weight
    
    # Binary cross-entropy loss
    return F.binary_cross_entropy(pred_adj.view(-1), target_adj.view(-1), weight=weight_tensor)


def evaluate_model(logreg, embeddings: torch.Tensor, 
                  labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Evaluate model performance on given data split.
    
    Args:
        logreg: Logistic regression classifier
        embeddings: Node embeddings
        labels: True labels
        mask: Data split mask
        
    Returns:
        float: Accuracy score
    """
    with torch.no_grad():
        logits = logreg(embeddings)
        predictions = torch.argmax(logits[mask], dim=1)
        accuracy = torch.sum(predictions == labels[mask]).float() / mask.sum()
    return accuracy.item()


def save_results(args, max_acc: float, max_test_acc: float) -> None:
    """
    Save experimental results to file.
    
    Args:
        args: Experiment arguments
        max_acc: Best validation accuracy
        max_test_acc: Corresponding test accuracy
    """
    result_path = './result'
    mkdir(result_path)
    
    result_file = os.path.join(result_path, 'GraphLoRA.txt')
    
    if args.few:
        result_line = (f'Few: True, r: {args.r}, Shot: {args.shot}, '
                      f'{args.pretrain_dataset} to {args.test_dataset}: '
                      f'val_acc: {max_acc:.6f}, test_acc: {max_test_acc:.6f}\n')
    else:
        result_line = (f'Few: False, r: {args.r}, '
                      f'{args.pretrain_dataset} to {args.test_dataset}: '
                      f'val_acc: {max_acc:.6f}, test_acc: {max_test_acc:.6f}\n')
    
    with open(result_file, 'a') as f:
        f.write(result_line)
    
    print(f"Results saved to: {result_file}")


def transfer(args, config: Dict[str, Any], gpu_id: int, is_reduction: bool = False) -> None:
    """
    Main transfer learning function for GraphLoRA.
    
    This function implements the complete GraphLoRA pipeline:
    1. Load and preprocess datasets
    2. Setup models (base GNN + LoRA adaptations)
    3. Create data splits and contrastive learning masks
    4. Train with multi-objective loss (classification + contrastive + reconstruction + domain adaptation)
    5. Evaluate and save results
    
    Args:
        args: Experiment configuration containing:
            - pretrain_dataset: Source domain dataset name
            - test_dataset: Target domain dataset name
            - r: LoRA rank for low-rank adaptation
            - few: Whether to use few-shot learning
            - shot: Number of shots per class (if few-shot)
            - num_epochs: Number of training epochs
            - lr1, lr2, lr3: Learning rates for different components
            - wd1, wd2, wd3: Weight decay values
            - l1, l2, l3, l4: Loss combination weights
            - tau: Temperature for contrastive loss
            - sup_weight: Supervision weight for contrastive mask
        config: Model configuration containing:
            - output_dim: GNN output dimension
            - activation: Activation function name
            - gnn_type: GNN architecture type
            - num_layers: Number of GNN layers
        gpu_id (int): GPU device ID
        is_reduction (bool): Whether to apply feature dimensionality reduction
        
    Returns:
        None
    """
    print("=" * 80)
    print("GRAPHLORA TRANSFER LEARNING")
    print("=" * 80)
    print(f"Source: {args.pretrain_dataset} -> Target: {args.test_dataset}")
    print(f"LoRA rank: {args.r}, Few-shot: {args.few}")
    if args.few:
        print(f"Shots per class: {args.shot}")
    
    # Setup device
    device = setup_device(gpu_id)
    
    # Load datasets
    pretrain_dataset, test_dataset = load_datasets(args, is_reduction)
    pretrain_dataset = pretrain_dataset.to(device)
    test_dataset = test_dataset.to(device)
    
    # Setup adjacency matrix for reconstruction loss
    target_adj = to_dense_adj(test_dataset.edge_index)[0]
    
    # Setup models
    gnn2, projector, logreg, SMMD = setup_models(args, config, pretrain_dataset, test_dataset, device)
    
    # Setup data splits
    train_mask, val_mask, test_mask, test_dataset = setup_data_splits(args, test_dataset, device)
    
    # Create contrastive learning mask and PPR weights
    mask = create_contrastive_mask(test_dataset, train_mask, args, device)
    ppr_weight = get_ppr_weight(test_dataset)
    
    # Setup optimizer with different learning rates for different components
    optimizer = torch.optim.Adam([
        {"params": projector.parameters(), 'lr': args.lr1, 'weight_decay': args.wd1},
        {"params": logreg.parameters(), 'lr': args.lr2, 'weight_decay': args.wd2},
        {"params": gnn2.parameters(), 'lr': args.lr3, 'weight_decay': args.wd3}
    ])
    
    print(f"\nOptimizer setup:")
    print(f"  Projector: lr={args.lr1}, wd={args.wd1}")
    print(f"  Classifier: lr={args.lr2}, wd={args.wd2}")
    print(f"  GNN LoRA: lr={args.lr3}, wd={args.wd3}")
    
    # Setup loss function and data loader
    loss_fn = nn.CrossEntropyLoss()
    pretrain_graph_loader = DataLoader(pretrain_dataset.x, batch_size=128, shuffle=True)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Loss weights: classification={args.l1}, domain_adapt={args.l2}, contrastive={args.l3}, reconstruction={args.l4}")
    
    max_acc = 0.0
    max_test_acc = 0.0
    max_epoch = 0
    
    for epoch in range(args.num_epochs):
        # Training phase
        logreg.train()
        projector.train()
        gnn2.train()
        
        # Get labels for current split
        train_labels = test_dataset.y[train_mask]
        val_labels = test_dataset.y[val_mask]
        test_labels = test_dataset.y[test_mask]
        
        # Forward pass
        optimizer.zero_grad()
        
        # Project target features to source domain
        feature_map = projector(test_dataset.x)
        
        # Get embeddings from LoRA-enhanced GNN
        emb, emb1, emb2 = gnn2(feature_map, test_dataset.edge_index)
        
        # Compute individual loss components
        
        # 1. Domain adaptation loss (SMMD)
        smmd_loss = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)
        
        # 2. Contrastive learning loss
        ct_loss = 0.5 * (
            batched_gct_loss(emb1, emb2, 1000, mask, args.tau) + 
            batched_gct_loss(emb2, emb1, 1000, mask, args.tau)
        ).mean()
        
        # 3. Classification loss
        logits = logreg(emb)
        train_logits = logits[train_mask]
        cls_loss = loss_fn(train_logits, train_labels)
        
        # 4. Graph reconstruction loss
        rec_loss = compute_reconstruction_loss(logits, target_adj, device)
        
        # Combined loss
        total_loss = (args.l1 * cls_loss + 
                     args.l2 * smmd_loss + 
                     args.l3 * ct_loss + 
                     args.l4 * rec_loss)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Compute training accuracy
        with torch.no_grad():
            train_preds = torch.argmax(train_logits, dim=1)
            train_acc = torch.sum(train_preds == train_labels).float() / train_labels.shape[0]
        
        # Evaluation phase
        logreg.eval()
        projector.eval()
        gnn2.eval()
        
        with torch.no_grad():
            val_acc = evaluate_model(logreg, emb, test_dataset.y, val_mask)
            test_acc = evaluate_model(logreg, emb, test_dataset.y, test_mask)
            
            # Print progress
            if epoch % 10 == 0 or epoch == args.num_epochs - 1:
                print(f'Epoch {epoch:3d}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, '
                      f'test_acc={test_acc:.4f}, loss={total_loss:.4f}')
                print(f'  └─ cls={cls_loss:.4f}, smmd={smmd_loss:.4f}, '
                      f'ct={ct_loss:.4f}, rec={rec_loss:.4f}')
            
            # Track best model
            if val_acc > max_acc:
                max_acc = val_acc
                max_test_acc = test_acc
                max_epoch = epoch + 1
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f'Best epoch: {max_epoch}')
    print(f'Best validation accuracy: {max_acc:.6f}')
    print(f'Corresponding test accuracy: {max_test_acc:.6f}')
    
    # Save results
    save_results(args, max_acc, max_test_acc)
