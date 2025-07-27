"""
Loss Functions for ENGINE-GraphLoRA

This module implements the comprehensive loss functions combining domain adaptation,
contrastive learning, and structure preservation for effective text-graph learning.

Key Components:
- Structure-aware Maximum Mean Discrepancy (SMMD) for domain alignment
- Contrastive text-structure loss for cross-modal consistency
- Graph reconstruction loss for structure preservation
- Multi-objective loss combination with adaptive weighting

Mathematical Foundations:
- SMMD: ||mean_S φ(z_v) - mean_T φ(z_u)||_H²
- Contrastive: InfoNCE loss between text and graph representations
- Reconstruction: Binary cross-entropy for adjacency prediction

Author: Graph Neural Network Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree, to_dense_adj
from typing import Optional, Dict, Union, Tuple, List
import logging

logger = logging.getLogger(__name__)


class StructureAwareMMD(nn.Module):
    """
    Structure-aware Maximum Mean Discrepancy (SMMD) loss for domain adaptation.
    
    This loss extends the standard MMD by incorporating structural information
    (degree, clustering coefficient) into the feature mapping, enabling better
    alignment between source and target graph domains.
    
    Mathematical formulation:
    φ(z) = concat(z, degree, log(1 + clustering_coef))
    L_SMMD = ||mean_S φ(z_v) - mean_T φ(z_u)||²_H
    
    Args:
        kernel_type (str): Kernel type ('rbf' or 'linear')
        kernel_mul (float): Kernel bandwidth multiplier for RBF
        kernel_num (int): Number of kernels for multi-scale RBF
        fix_sigma (Optional[float]): Fixed bandwidth (if not adaptive)
        include_structure (bool): Whether to include structural features
        structure_weight (float): Weight for structural features
    """
    
    def __init__(
        self,
        kernel_type: str = 'rbf',
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: Optional[float] = None,
        include_structure: bool = True,
        structure_weight: float = 0.1
    ):
        super(StructureAwareMMD, self).__init__()
        
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.include_structure = include_structure
        self.structure_weight = structure_weight
        
        logger.info(f"SMMD initialized: kernel={kernel_type}, structure={include_structure}")
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_edge_index: Optional[torch.Tensor] = None,
        target_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute structure-aware MMD loss between source and target domains.
        
        Args:
            source_features (torch.Tensor): Source domain features [n_source, d]
            target_features (torch.Tensor): Target domain features [n_target, d]
            source_edge_index (Optional[torch.Tensor]): Source graph edges
            target_edge_index (Optional[torch.Tensor]): Target graph edges
            
        Returns:
            torch.Tensor: SMMD loss scalar
        """
        # Augment features with structural information
        if self.include_structure:
            source_features = self._augment_with_structure(source_features, source_edge_index)
            target_features = self._augment_with_structure(target_features, target_edge_index)
        
        # Compute MMD between augmented features
        return self._compute_mmd(source_features, target_features)
    
    def _augment_with_structure(
        self, 
        features: torch.Tensor, 
        edge_index: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Augment node features with structural information.
        
        Args:
            features (torch.Tensor): Node features [num_nodes, feature_dim]
            edge_index (Optional[torch.Tensor]): Graph edges [2, num_edges]
            
        Returns:
            torch.Tensor: Augmented features [num_nodes, feature_dim + structure_dim]
        """
        if edge_index is None:
            # If no graph structure, pad with zeros
            num_nodes = features.size(0)
            structure_features = torch.zeros(num_nodes, 2, device=features.device)
        else:
            # Compute structural features
            num_nodes = features.size(0)
            
            # 1. Degree centrality
            node_degrees = degree(edge_index[1], num_nodes, dtype=torch.float)
            normalized_degrees = node_degrees / (num_nodes - 1) if num_nodes > 1 else node_degrees
            
            # 2. Clustering coefficient (simplified approximation)
            clustering_coef = self._compute_clustering_coefficient(edge_index, num_nodes)
            log_clustering = torch.log(1 + clustering_coef)
            
            # Combine structural features
            structure_features = torch.stack([normalized_degrees, log_clustering], dim=1)
        
        # Concatenate with original features
        augmented_features = torch.cat([
            features, 
            self.structure_weight * structure_features
        ], dim=1)
        
        return augmented_features
    
    def _compute_clustering_coefficient(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute local clustering coefficient for each node.
        
        Args:
            edge_index (torch.Tensor): Graph edges [2, num_edges]
            num_nodes (int): Number of nodes
            
        Returns:
            torch.Tensor: Clustering coefficients [num_nodes]
        """
        device = edge_index.device
        clustering = torch.zeros(num_nodes, device=device)
        
        # Create adjacency matrix for efficient computation
        adj_matrix = to_dense_adj(edge_index)[0]  # [num_nodes, num_nodes]
        
        for node in range(num_nodes):
            # Find neighbors
            neighbors = torch.nonzero(adj_matrix[node], as_tuple=True)[0]
            degree_node = len(neighbors)
            
            if degree_node < 2:
                clustering[node] = 0.0
            else:
                # Count triangles (edges between neighbors)
                neighbor_adj = adj_matrix[neighbors][:, neighbors]
                triangles = torch.sum(neighbor_adj) / 2  # Each edge counted twice
                
                # Clustering coefficient = 2 * triangles / (degree * (degree - 1))
                max_edges = degree_node * (degree_node - 1) / 2
                clustering[node] = triangles / max_edges if max_edges > 0 else 0.0
        
        return clustering
    
    def _compute_mmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy between two distributions.
        
        Args:
            X (torch.Tensor): Source samples [n_x, d]
            Y (torch.Tensor): Target samples [n_y, d]
            
        Returns:
            torch.Tensor: MMD² scalar
        """
        if self.kernel_type == 'linear':
            return self._linear_mmd(X, Y)
        elif self.kernel_type == 'rbf':
            return self._rbf_mmd(X, Y)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _linear_mmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear MMD."""
        mean_X = torch.mean(X, dim=0)
        mean_Y = torch.mean(Y, dim=0)
        return torch.norm(mean_X - mean_Y, p=2) ** 2
    
    def _rbf_mmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF MMD with multiple bandwidths.
        
        Args:
            X (torch.Tensor): Source samples [n_x, d]
            Y (torch.Tensor): Target samples [n_y, d]
            
        Returns:
            torch.Tensor: RBF MMD² scalar
        """
        n_x, n_y = X.size(0), Y.size(0)
        
        # Concatenate samples
        XY = torch.cat([X, Y], dim=0)
        
        # Compute pairwise distances
        dists = torch.cdist(XY, XY, p=2) ** 2
        
        # Compute bandwidth if not fixed
        if self.fix_sigma is None:
            # Use median heuristic
            triu_indices = torch.triu_indices(XY.size(0), XY.size(0), offset=1)
            sigma = torch.median(dists[triu_indices[0], triu_indices[1]])
        else:
            sigma = self.fix_sigma
        
        # Multi-scale RBF kernels
        kernel_values = torch.zeros_like(dists)
        for i in range(self.kernel_num):
            bandwidth = sigma * (self.kernel_mul ** (i - self.kernel_num // 2))
            kernel_values += torch.exp(-dists / (2 * bandwidth))
        
        kernel_values = kernel_values / self.kernel_num
        
        # Compute MMD² terms
        # K_XX
        K_XX = kernel_values[:n_x, :n_x]
        K_XX_mean = (torch.sum(K_XX) - torch.trace(K_XX)) / (n_x * (n_x - 1))
        
        # K_YY  
        K_YY = kernel_values[n_x:, n_x:]
        K_YY_mean = (torch.sum(K_YY) - torch.trace(K_YY)) / (n_y * (n_y - 1))
        
        # K_XY
        K_XY = kernel_values[:n_x, n_x:]
        K_XY_mean = torch.mean(K_XY)
        
        # MMD² = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
        mmd_squared = K_XX_mean + K_YY_mean - 2 * K_XY_mean
        
        return torch.clamp(mmd_squared, min=0.0)  # Ensure non-negative


class ContrastiveTextStructureLoss(nn.Module):
    """
    Contrastive loss for aligning text and graph representations.
    
    This loss encourages agreement between G-Ladder (text-aware) outputs and
    LoRA-SAGE (structure-aware) representations for the same nodes, while
    pushing apart representations of different nodes.
    
    Uses InfoNCE loss:
    L = -log(exp(sim(h_text, h_graph) / τ) / Σ_k exp(sim(h_text, h_k) / τ))
    
    Args:
        temperature (float): Temperature parameter for softmax
        similarity_metric (str): Similarity metric ('cosine' or 'dot')
        negative_sampling (str): Negative sampling strategy ('random' or 'hard')
        num_negatives (int): Number of negative samples per positive
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        similarity_metric: str = 'cosine',
        negative_sampling: str = 'random',
        num_negatives: int = 64
    ):
        super(ContrastiveTextStructureLoss, self).__init__()
        
        self.temperature = temperature
        self.similarity_metric = similarity_metric
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives
        
        logger.info(f"Contrastive loss initialized: τ={temperature}, negatives={num_negatives}")
    
    def forward(
        self,
        text_representations: torch.Tensor,
        graph_representations: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between text and graph representations.
        
        Args:
            text_representations (torch.Tensor): Text-derived features [num_nodes, d]
            graph_representations (torch.Tensor): Graph-derived features [num_nodes, d]
            node_mask (Optional[torch.Tensor]): Mask for valid nodes [num_nodes]
            
        Returns:
            torch.Tensor: Contrastive loss scalar
        """
        if node_mask is not None:
            text_representations = text_representations[node_mask]
            graph_representations = graph_representations[node_mask]
        
        # Normalize representations
        text_norm = F.normalize(text_representations, p=2, dim=1)
        graph_norm = F.normalize(graph_representations, p=2, dim=1)
        
        # Compute InfoNCE loss in both directions
        loss_text_to_graph = self._info_nce_loss(text_norm, graph_norm)
        loss_graph_to_text = self._info_nce_loss(graph_norm, text_norm)
        
        # Symmetric loss
        total_loss = 0.5 * (loss_text_to_graph + loss_graph_to_text)
        
        return total_loss
    
    def _info_nce_loss(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss between query and key representations.
        
        Args:
            query (torch.Tensor): Query representations [batch_size, d]
            key (torch.Tensor): Key representations [batch_size, d]
            
        Returns:
            torch.Tensor: InfoNCE loss
        """
        batch_size = query.size(0)
        
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            similarities = torch.mm(query, key.t())  # [batch_size, batch_size]
        elif self.similarity_metric == 'dot':
            # Dot product similarity
            similarities = torch.mm(query, key.t())
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Scale by temperature
        similarities = similarities / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=query.device)
        
        # InfoNCE loss (cross-entropy with diagonal as positive)
        loss = F.cross_entropy(similarities, labels)
        
        return loss


class GraphReconstructionLoss(nn.Module):
    """
    Graph reconstruction loss for structure preservation.
    
    This loss encourages the model to preserve graph structure information
    by predicting adjacency relationships from learned node representations.
    
    Args:
        reconstruction_type (str): Type of reconstruction ('adjacency' or 'edge_features')
        pos_weight_factor (float): Weight factor for positive edges (to handle imbalance)
        use_negative_sampling (bool): Whether to use negative edge sampling
        negative_sampling_ratio (float): Ratio of negative to positive edges
    """
    
    def __init__(
        self,
        reconstruction_type: str = 'adjacency',
        pos_weight_factor: float = 1.0,
        use_negative_sampling: bool = True,
        negative_sampling_ratio: float = 1.0
    ):
        super(GraphReconstructionLoss, self).__init__()
        
        self.reconstruction_type = reconstruction_type
        self.pos_weight_factor = pos_weight_factor
        self.use_negative_sampling = use_negative_sampling
        self.negative_sampling_ratio = negative_sampling_ratio
        
        logger.info(f"Reconstruction loss initialized: type={reconstruction_type}")
    
    def forward(
        self,
        node_representations: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute graph reconstruction loss.
        
        Args:
            node_representations (torch.Tensor): Node embeddings [num_nodes, d]
            edge_index (torch.Tensor): True graph edges [2, num_edges]
            num_nodes (Optional[int]): Number of nodes (inferred if None)
            
        Returns:
            torch.Tensor: Reconstruction loss scalar
        """
        if num_nodes is None:
            num_nodes = node_representations.size(0)
        
        # Predict edge probabilities
        edge_probs = self._predict_edges(node_representations, edge_index, num_nodes)
        
        # Create target adjacency
        target_adj = self._create_target_adjacency(edge_index, num_nodes)
        
        # Compute weighted binary cross-entropy
        return self._compute_reconstruction_loss(edge_probs, target_adj)
    
    def _predict_edges(
        self,
        node_representations: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Predict edge probabilities from node representations.
        
        Args:
            node_representations (torch.Tensor): Node embeddings [num_nodes, d]
            edge_index (torch.Tensor): Edge indices for prediction [2, num_edges]
            num_nodes (int): Number of nodes
            
        Returns:
            torch.Tensor: Edge probabilities [num_nodes, num_nodes]
        """
        # Compute pairwise similarities (dot product)
        similarities = torch.mm(node_representations, node_representations.t())
        
        # Apply sigmoid to get probabilities
        edge_probs = torch.sigmoid(similarities)
        
        return edge_probs
    
    def _create_target_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Create target adjacency matrix from edge indices.
        
        Args:
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            num_nodes (int): Number of nodes
            
        Returns:
            torch.Tensor: Target adjacency matrix [num_nodes, num_nodes]
        """
        device = edge_index.device
        target_adj = torch.zeros(num_nodes, num_nodes, device=device)
        
        # Set edges to 1
        target_adj[edge_index[0], edge_index[1]] = 1.0
        
        # Make symmetric for undirected graphs
        target_adj = torch.max(target_adj, target_adj.t())
        
        return target_adj
    
    def _compute_reconstruction_loss(
        self,
        predicted_adj: torch.Tensor,
        target_adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted binary cross-entropy for reconstruction.
        
        Args:
            predicted_adj (torch.Tensor): Predicted adjacency [num_nodes, num_nodes]
            target_adj (torch.Tensor): Target adjacency [num_nodes, num_nodes]
            
        Returns:
            torch.Tensor: Reconstruction loss scalar
        """
        # Flatten matrices
        pred_flat = predicted_adj.view(-1)
        target_flat = target_adj.view(-1)
        
        # Compute positive weight for class imbalance
        num_pos = target_flat.sum()
        num_neg = len(target_flat) - num_pos
        pos_weight = (num_neg / num_pos) * self.pos_weight_factor if num_pos > 0 else 1.0
        
        # Weighted binary cross-entropy
        weight_mask = target_flat * pos_weight + (1 - target_flat)
        bce_loss = F.binary_cross_entropy(pred_flat, target_flat, weight=weight_mask)
        
        return bce_loss


class MultiObjectiveLoss(nn.Module):
    """
    Combined multi-objective loss for ENGINE-GraphLoRA training.
    
    This loss combines multiple objectives with adaptive or fixed weighting:
    - Classification loss (primary task)
    - SMMD loss (domain adaptation)
    - Contrastive loss (cross-modal alignment)  
    - Reconstruction loss (structure preservation)
    
    Args:
        loss_weights (Dict[str, float]): Fixed weights for each loss component
        adaptive_weighting (bool): Whether to use adaptive loss weighting
        uncertainty_weighting (bool): Whether to use uncertainty-based weighting
    """
    
    def __init__(
        self,
        loss_weights: Dict[str, float] = {
            'classification': 1.0,
            'smmd': 0.1,
            'contrastive': 0.1,
            'reconstruction': 0.01
        },
        adaptive_weighting: bool = False,
        uncertainty_weighting: bool = False
    ):
        super(MultiObjectiveLoss, self).__init__()
        
        self.loss_weights = loss_weights
        self.adaptive_weighting = adaptive_weighting
        self.uncertainty_weighting = uncertainty_weighting
        
        # Initialize individual loss modules
        self.smmd_loss = StructureAwareMMD()
        self.contrastive_loss = ContrastiveTextStructureLoss()
        self.reconstruction_loss = GraphReconstructionLoss()
        
        # Learnable uncertainty parameters (if using uncertainty weighting)
        if uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                'classification': nn.Parameter(torch.zeros(1)),
                'smmd': nn.Parameter(torch.zeros(1)),
                'contrastive': nn.Parameter(torch.zeros(1)),
                'reconstruction': nn.Parameter(torch.zeros(1))
            })
        
        # Running averages for adaptive weighting
        if adaptive_weighting:
            self.register_buffer('loss_history', torch.zeros(4, 100))  # 4 losses, 100 history
            self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
        logger.info(f"Multi-objective loss initialized: weights={loss_weights}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        text_representations: torch.Tensor,
        graph_representations: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_index: torch.Tensor,
        source_edge_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined multi-objective loss.
        
        Args:
            predictions (torch.Tensor): Model predictions for classification
            targets (torch.Tensor): Ground truth labels
            text_representations (torch.Tensor): Text-derived representations
            graph_representations (torch.Tensor): Graph-derived representations
            source_features (torch.Tensor): Source domain features
            target_features (torch.Tensor): Target domain features
            edge_index (torch.Tensor): Target graph edges
            source_edge_index (Optional[torch.Tensor]): Source graph edges
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: (total_loss, individual_losses)
        """
        individual_losses = {}
        
        # 1. Classification loss
        classification_loss = F.cross_entropy(predictions, targets)
        individual_losses['classification'] = classification_loss
        
        # 2. SMMD loss (domain adaptation)
        smmd_loss = self.smmd_loss(
            source_features, target_features,
            source_edge_index, edge_index
        )
        individual_losses['smmd'] = smmd_loss
        
        # 3. Contrastive loss (cross-modal alignment)
        contrastive_loss = self.contrastive_loss(
            text_representations, graph_representations
        )
        individual_losses['contrastive'] = contrastive_loss
        
        # 4. Reconstruction loss (structure preservation)
        reconstruction_loss = self.reconstruction_loss(
            graph_representations, edge_index
        )
        individual_losses['reconstruction'] = reconstruction_loss
        
        # Combine losses with appropriate weighting
        if self.uncertainty_weighting:
            total_loss = self._uncertainty_weighted_combination(individual_losses)
        elif self.adaptive_weighting:
            total_loss = self._adaptive_weighted_combination(individual_losses)
        else:
            total_loss = self._fixed_weighted_combination(individual_losses)
        
        return total_loss, individual_losses
    
    def _fixed_weighted_combination(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine losses with fixed weights."""
        total_loss = torch.tensor(0.0, device=list(losses.values())[0].device)
        
        for loss_name, loss_value in losses.items():
            weight = self.loss_weights.get(loss_name, 0.0)
            total_loss += weight * loss_value
        
        return total_loss
    
    def _uncertainty_weighted_combination(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine losses using uncertainty-based weighting.
        
        Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
        Loss = Σ (1/(2σ²)) * L_i + log(σ)
        """
        total_loss = torch.tensor(0.0, device=list(losses.values())[0].device)
        
        for loss_name, loss_value in losses.items():
            if loss_name in self.log_vars:
                sigma_squared = torch.exp(self.log_vars[loss_name])
                weighted_loss = (1.0 / (2 * sigma_squared)) * loss_value + 0.5 * self.log_vars[loss_name]
                total_loss += weighted_loss
        
        return total_loss
    
    def _adaptive_weighted_combination(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine losses using adaptive weighting based on loss magnitude history.
        """
        # Update loss history
        self._update_loss_history(losses)
        
        # Compute adaptive weights based on relative loss magnitudes
        weights = self._compute_adaptive_weights()
        
        total_loss = torch.tensor(0.0, device=list(losses.values())[0].device)
        loss_names = ['classification', 'smmd', 'contrastive', 'reconstruction']
        
        for i, loss_name in enumerate(loss_names):
            if loss_name in losses:
                total_loss += weights[i] * losses[loss_name]
        
        return total_loss
    
    def _update_loss_history(self, losses: Dict[str, torch.Tensor]):
        """Update running history of loss values."""
        with torch.no_grad():
            ptr = self.history_ptr.item()
            loss_names = ['classification', 'smmd', 'contrastive', 'reconstruction']
            
            for i, loss_name in enumerate(loss_names):
                if loss_name in losses:
                    self.loss_history[i, ptr] = losses[loss_name].item()
            
            self.history_ptr[0] = (ptr + 1) % self.loss_history.size(1)
    
    def _compute_adaptive_weights(self) -> torch.Tensor:
        """Compute adaptive weights based on loss history."""
        # Compute running averages
        valid_history = self.loss_history[:, self.loss_history[0] != 0]
        
        if valid_history.size(1) > 0:
            avg_losses = valid_history.mean(dim=1)
            # Inverse weighting: higher loss magnitude gets lower weight
            weights = 1.0 / (avg_losses + 1e-8)
            weights = weights / weights.sum()  # Normalize
        else:
            # Fallback to equal weighting
            weights = torch.ones(4) / 4
        
        return weights.to(self.loss_history.device)
    
    def get_loss_stats(self) -> Dict[str, float]:
        """Get statistics about loss components."""
        stats = {'weights': self.loss_weights.copy()}
        
        if self.uncertainty_weighting:
            stats['uncertainties'] = {
                name: torch.exp(param).item() 
                for name, param in self.log_vars.items()
            }
        
        if self.adaptive_weighting:
            adaptive_weights = self._compute_adaptive_weights()
            loss_names = ['classification', 'smmd', 'contrastive', 'reconstruction']
            stats['adaptive_weights'] = {
                name: weight.item() 
                for name, weight in zip(loss_names, adaptive_weights)
            }
        
        return stats 