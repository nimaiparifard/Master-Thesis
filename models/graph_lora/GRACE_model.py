"""
GRACE: Graph Random neural network with Adaptive Contrast Enhancement

This module implements the GRACE model for self-supervised graph representation learning
using contrastive learning. GRACE uses data augmentation (edge dropping and feature masking)
to create different views of the same graph and learns representations by maximizing
agreement between positive pairs while minimizing agreement with negative pairs.

Key Features:
- Graph contrastive learning with InfoNCE loss
- Dual data augmentation (edge dropping + feature masking)  
- Projection head for contrastive learning
- Batch processing for memory efficiency
- Temperature-scaled similarity

Reference:
    Zhu, Y., et al. "Deep Graph Contrastive Representation Learning." 
    arXiv preprint arXiv:2006.04131 (2020).

Author: Graph LoRA Team
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from typing import Tuple, Optional
import math


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """
    Randomly drop (mask to zero) node features for data augmentation.
    
    This function implements feature dropping as a form of data augmentation
    for graph contrastive learning. It randomly selects feature dimensions
    and sets them to zero across all nodes.
    
    Args:
        x (torch.Tensor): Node feature matrix [num_nodes, num_features]
        drop_prob (float): Probability of dropping each feature dimension
        
    Returns:
        torch.Tensor: Feature matrix with randomly dropped features
        
    Note:
        - Features are dropped dimension-wise (affects all nodes equally)
        - Original tensor is cloned to avoid in-place modification
        - Dropout is applied consistently across the entire batch
    """
    if drop_prob <= 0.0:
        return x
    
    # Generate random mask for feature dimensions
    drop_mask = torch.empty(
        (x.size(1),), 
        dtype=torch.float32,
        device=x.device
    ).uniform_(0, 1) < drop_prob
    
    # Clone to avoid modifying original tensor
    x_dropped = x.clone()
    x_dropped[:, drop_mask] = 0.0
    
    return x_dropped


class GRACE(torch.nn.Module):
    """
    GRACE (Graph Random neural network with Adaptive Contrast Enhancement) model.
    
    This model implements self-supervised graph representation learning using
    contrastive learning with graph data augmentation. It creates two augmented
    views of the input graph and learns representations by maximizing agreement
    between corresponding nodes while minimizing agreement with other nodes.
    
    Args:
        gnn (torch.nn.Module): Graph neural network encoder
        num_hidden (int): Hidden dimension of GNN output
        num_proj_hidden (int): Hidden dimension of projection head
        drop_edge_rate (float): Probability of dropping edges (0.0-1.0)
        drop_feature_rate (float): Probability of dropping features (0.0-1.0)
        tau (float): Temperature parameter for contrastive loss (typically 0.1-1.0)
        
    Attributes:
        gnn: Graph neural network encoder
        tau: Temperature parameter for InfoNCE loss
        drop_edge_rate: Edge dropout probability
        drop_feature_rate: Feature dropout probability
        fc1: First layer of projection head
        fc2: Second layer of projection head
    """
    
    def __init__(self, 
                 gnn: torch.nn.Module, 
                 num_hidden: int, 
                 num_proj_hidden: int, 
                 drop_edge_rate: float, 
                 drop_feature_rate: float, 
                 tau: float = 0.5):
        super(GRACE, self).__init__()
        
        # Validate input parameters
        if not 0.0 <= drop_edge_rate <= 1.0:
            raise ValueError("drop_edge_rate must be between 0.0 and 1.0")
        if not 0.0 <= drop_feature_rate <= 1.0:
            raise ValueError("drop_feature_rate must be between 0.0 and 1.0")
        if tau <= 0.0:
            raise ValueError("tau (temperature) must be positive")
        if num_hidden <= 0 or num_proj_hidden <= 0:
            raise ValueError("Hidden dimensions must be positive")
        
        self.gnn = gnn
        self.tau = tau
        self.drop_edge_rate = drop_edge_rate
        self.drop_feature_rate = drop_feature_rate

        # Projection head for contrastive learning
        # Two-layer MLP: hidden -> proj_hidden -> hidden
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        
        # Initialize projection head weights
        self._init_projection_head()

    def _init_projection_head(self) -> None:
        """Initialize projection head weights using Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            torch.nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN encoder.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, num_hidden]
        """
        return self.gnn(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply projection head to node embeddings.
        
        The projection head maps node embeddings to a space where
        contrastive learning is performed. This helps separate the
        representation learning from the contrastive objective.
        
        Args:
            z (torch.Tensor): Node embeddings [num_nodes, num_hidden]
            
        Returns:
            torch.Tensor: Projected embeddings [num_nodes, num_hidden]
        """
        # Two-layer projection with ELU activation
        h = F.elu(self.fc1(z))
        return self.fc2(h)

    def compute_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            z1 (torch.Tensor): First set of embeddings [N, D]
            z2 (torch.Tensor): Second set of embeddings [M, D]
            
        Returns:
            torch.Tensor: Cosine similarity matrix [N, M]
        """
        # L2 normalize embeddings
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        
        # Compute cosine similarity
        return torch.mm(z1_norm, z2_norm.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE (contrastive) loss for a batch of embeddings.
        
        This function implements the InfoNCE loss where each embedding in z1
        is considered as an anchor, its corresponding embedding in z2 as positive,
        and all other embeddings as negatives.
        
        Args:
            z1 (torch.Tensor): First view embeddings [N, D]
            z2 (torch.Tensor): Second view embeddings [N, D]
            
        Returns:
            torch.Tensor: InfoNCE loss for each sample [N]
        """
        # Temperature-scaled exponential function
        f = lambda x: torch.exp(x / self.tau)
        
        # Compute similarity matrices
        refl_sim = f(self.compute_similarity(z1, z1))    # [N, N] - within view similarity
        between_sim = f(self.compute_similarity(z1, z2)) # [N, N] - cross view similarity

        # InfoNCE loss computation
        # Numerator: positive pairs (diagonal of between_sim)
        # Denominator: all pairs except self-similarity
        numerator = between_sim.diag()
        denominator = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        
        # Avoid division by zero and numerical instability
        denominator = torch.clamp(denominator, min=1e-8)
        
        return -torch.log(numerator / denominator)

    def batched_semi_loss(self, 
                         z1: torch.Tensor, 
                         z2: torch.Tensor,
                         batch_size: int) -> torch.Tensor:
        """
        Compute InfoNCE loss in batches for memory efficiency.
        
        For large graphs, computing the full similarity matrix can be memory-intensive.
        This function processes the loss computation in smaller batches to reduce
        memory usage while maintaining the same loss computation.
        
        Args:
            z1 (torch.Tensor): First view embeddings [N, D]
            z2 (torch.Tensor): Second view embeddings [N, D]
            batch_size (int): Size of processing batches
            
        Returns:
            torch.Tensor: InfoNCE loss for each sample [N]
        """
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = math.ceil(num_nodes / batch_size)
        f = lambda x: torch.exp(x / self.tau)
        
        losses = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_nodes)
            batch_indices = torch.arange(start_idx, end_idx, device=device)
            
            # Get current batch
            z1_batch = z1[batch_indices]  # [B, D]
            
            # Compute similarities for current batch against all nodes
            refl_sim = f(self.compute_similarity(z1_batch, z1))    # [B, N]
            between_sim = f(self.compute_similarity(z1_batch, z2)) # [B, N]

            # Extract positive pairs for current batch
            positive_sim = between_sim[:, start_idx:end_idx].diag()  # [B]
            
            # Compute denominators (exclude self-similarities)
            self_sim = refl_sim[:, start_idx:end_idx].diag()  # [B]
            denominator = refl_sim.sum(1) + between_sim.sum(1) - self_sim
            
            # Avoid numerical instability
            denominator = torch.clamp(denominator, min=1e-8)
            
            # Compute batch loss
            batch_loss = -torch.log(positive_sim / denominator)
            losses.append(batch_loss)

        return torch.cat(losses)

    def loss(self, 
            z1: torch.Tensor, 
            z2: torch.Tensor,
            mean: bool = True, 
            batch_size: int = 0) -> torch.Tensor:
        """
        Compute symmetric contrastive loss between two views.
        
        This function computes the full contrastive loss by considering both
        directions: z1->z2 and z2->z1, then averaging them for symmetry.
        
        Args:
            z1 (torch.Tensor): First view embeddings [N, D]
            z2 (torch.Tensor): Second view embeddings [N, D]
            mean (bool): Whether to return mean loss or sum
            batch_size (int): Batch size for memory-efficient computation (0 = no batching)
            
        Returns:
            torch.Tensor: Contrastive loss (scalar)
        """
        # Apply projection heads
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        # Compute bidirectional loss
        if batch_size <= 0:
            # Standard computation (may use more memory)
            l1 = self.semi_loss(h1, h2)  # h1 as anchor
            l2 = self.semi_loss(h2, h1)  # h2 as anchor
        else:
            # Memory-efficient batched computation
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        # Symmetric loss
        loss = (l1 + l2) * 0.5
        
        # Return mean or sum
        return loss.mean() if mean else loss.sum()

    def create_augmented_views(self, 
                              x: torch.Tensor, 
                              edge_index: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                                                Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create two augmented views of the input graph.
        
        This function applies different data augmentation strategies to create
        two correlated but distinct views of the same graph for contrastive learning.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            Tuple containing two views, each as (features, edge_index):
                - First view: (x1, edge_index1)  
                - Second view: (x2, edge_index2)
        """
        # Create first augmented view
        edge_index_1 = dropout_edge(edge_index, p=self.drop_edge_rate)[0]
        x_1 = drop_feature(x, self.drop_feature_rate)
        
        # Create second augmented view (independent augmentation)
        edge_index_2 = dropout_edge(edge_index, p=self.drop_edge_rate)[0]
        x_2 = drop_feature(x, self.drop_feature_rate)
        
        return (x_1, edge_index_1), (x_2, edge_index_2)

    def compute_loss(self, 
                    x: torch.Tensor, 
                    edge_index: torch.Tensor, 
                    batch_size: int = 1000) -> torch.Tensor:
        """
        Compute the complete GRACE contrastive loss.
        
        This is the main function for training GRACE. It creates two augmented
        views of the input graph, encodes them using the GNN, and computes
        the contrastive loss between the resulting embeddings.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            batch_size (int): Batch size for memory-efficient loss computation
            
        Returns:
            torch.Tensor: Contrastive loss (scalar)
        """
        # Create augmented views
        (x_1, edge_index_1), (x_2, edge_index_2) = self.create_augmented_views(x, edge_index)
        
        # Encode both views
        z1 = self.forward(x_1, edge_index_1)
        z2 = self.forward(x_2, edge_index_2)

        # Compute contrastive loss
        return self.loss(z1, z2, batch_size=batch_size)

    def get_embeddings(self, 
                      x: torch.Tensor, 
                      edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings without augmentation (for evaluation/inference).
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, num_hidden]
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x, edge_index)
        return embeddings

    def __repr__(self) -> str:
        """String representation of the GRACE model."""
        return (f"GRACE(tau={self.tau}, "
                f"drop_edge_rate={self.drop_edge_rate}, "
                f"drop_feature_rate={self.drop_feature_rate})")
