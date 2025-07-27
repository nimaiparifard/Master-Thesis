"""
Cross-Modal Fusion and Early-Exit Inference

This module implements the cross-modal fusion mechanisms and early-exit inference
strategies for ENGINE-GraphLoRA. It combines text and graph representations and
provides dynamic computation control through entropy-based early stopping.

Key Features:
- Bi-affine cross-modal fusion projectors
- Multi-scale attention-based fusion
- Early-exit inference with entropy thresholding
- Dynamic layer-wise computation control
- Adaptive confidence scoring

Author: Graph Neural Network Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
import logging

logger = logging.getLogger(__name__)


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion module for combining text and graph representations.
    
    This module implements various fusion strategies including bi-affine projections,
    attention-based fusion, and multi-scale combination for effective text-graph
    integration at strategic depths in the model.
    
    Mathematical formulation:
    c_v = ReLU(W1 · h_v + W2 · z_v + b)
    
    Where h_v is text representation and z_v is graph representation.
    
    Args:
        text_dim (int): Dimension of text representations
        graph_dim (int): Dimension of graph representations
        hidden_dim (int): Hidden dimension for fusion
        output_dim (int): Output dimension after fusion
        fusion_type (str): Fusion strategy ('biaffine', 'attention', 'concat', 'add')
        num_fusion_layers (int): Number of fusion layers to stack
        dropout (float): Dropout rate for regularization
        layer_norm (bool): Whether to apply layer normalization
    """
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        hidden_dim: int,
        output_dim: int,
        fusion_type: str = 'biaffine',
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super(CrossModalFusion, self).__init__()
        
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.num_fusion_layers = num_fusion_layers
        
        # Build fusion layers based on strategy
        if fusion_type == 'biaffine':
            self.fusion_layers = self._build_biaffine_fusion(dropout, layer_norm)
        elif fusion_type == 'attention':
            self.fusion_layers = self._build_attention_fusion(dropout, layer_norm)
        elif fusion_type == 'concat':
            self.fusion_layers = self._build_concat_fusion(dropout, layer_norm)
        elif fusion_type == 'add':
            self.fusion_layers = self._build_additive_fusion(dropout, layer_norm)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        logger.info(f"CrossModalFusion initialized: {fusion_type}, {text_dim}+{graph_dim}->{output_dim}")
    
    def forward(
        self,
        text_features: torch.Tensor,
        graph_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse text and graph representations.
        
        Args:
            text_features (torch.Tensor): Text representations [batch_size, text_dim]
            graph_features (torch.Tensor): Graph representations [batch_size, graph_dim]
            attention_mask (Optional[torch.Tensor]): Attention mask for valid features
            
        Returns:
            torch.Tensor: Fused representations [batch_size, output_dim]
        """
        return self.fusion_layers(text_features, graph_features, attention_mask)
    
    def _build_biaffine_fusion(self, dropout: float, layer_norm: bool) -> nn.Module:
        """Build bi-affine fusion layers."""
        return BiAffineFusion(
            self.text_dim, self.graph_dim, self.hidden_dim, self.output_dim,
            self.num_fusion_layers, dropout, layer_norm
        )
    
    def _build_attention_fusion(self, dropout: float, layer_norm: bool) -> nn.Module:
        """Build attention-based fusion layers."""
        return AttentionFusion(
            self.text_dim, self.graph_dim, self.hidden_dim, self.output_dim,
            self.num_fusion_layers, dropout, layer_norm
        )
    
    def _build_concat_fusion(self, dropout: float, layer_norm: bool) -> nn.Module:
        """Build concatenation-based fusion layers."""
        return ConcatFusion(
            self.text_dim, self.graph_dim, self.hidden_dim, self.output_dim,
            self.num_fusion_layers, dropout, layer_norm
        )
    
    def _build_additive_fusion(self, dropout: float, layer_norm: bool) -> nn.Module:
        """Build additive fusion layers."""
        return AdditiveFusion(
            self.text_dim, self.graph_dim, self.hidden_dim, self.output_dim,
            self.num_fusion_layers, dropout, layer_norm
        )


class BiAffineFusion(nn.Module):
    """
    Bi-affine fusion using separate linear projections for text and graph features.
    
    This implements the fusion strategy described in the ENGINE paper with
    learnable linear transformations for each modality.
    """
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super(BiAffineFusion, self).__init__()
        
        # Text and graph projection layers
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        
        # Fusion MLP layers
        fusion_layers = []
        current_dim = hidden_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer to output dimension
                fusion_layers.append(nn.Linear(current_dim, output_dim))
            else:
                # Hidden layers
                fusion_layers.append(nn.Linear(current_dim, hidden_dim))
                if layer_norm:
                    fusion_layers.append(nn.LayerNorm(hidden_dim))
                fusion_layers.append(nn.ReLU())
                fusion_layers.append(nn.Dropout(dropout))
        
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        
        # Initialize weights
        self._init_weights()
    
    def forward(
        self,
        text_features: torch.Tensor,
        graph_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through bi-affine fusion."""
        # Project both modalities to same dimension
        text_proj = self.text_proj(text_features)
        graph_proj = self.graph_proj(graph_features)
        
        # Element-wise combination
        combined = text_proj + graph_proj  # Can also use concat or other combinations
        
        # Apply fusion MLP
        fused = self.fusion_mlp(combined)
        
        return fused
    
    def _init_weights(self):
        """Initialize fusion weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class AttentionFusion(nn.Module):
    """
    Attention-based cross-modal fusion.
    
    Uses cross-attention mechanisms to allow text and graph representations
    to attend to each other for more sophisticated fusion.
    """
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super(AttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project to common dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        
        # Cross-attention layers
        self.text_to_graph_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.graph_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.layer_norm2 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        
        self._init_weights()
    
    def forward(
        self,
        text_features: torch.Tensor,
        graph_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through attention-based fusion."""
        # Project to common dimension
        text_proj = self.text_proj(text_features).unsqueeze(1)  # Add sequence dimension
        graph_proj = self.graph_proj(graph_features).unsqueeze(1)
        
        # Cross-attention: text attends to graph
        text_attended, _ = self.text_to_graph_attn(
            text_proj, graph_proj, graph_proj
        )
        text_attended = self.layer_norm1(text_attended.squeeze(1))
        
        # Cross-attention: graph attends to text
        graph_attended, _ = self.graph_to_text_attn(
            graph_proj, text_proj, text_proj
        )
        graph_attended = self.layer_norm2(graph_attended.squeeze(1))
        
        # Concatenate attended representations
        fused = torch.cat([text_attended, graph_attended], dim=-1)
        
        # Final projection
        output = self.output_proj(fused)
        
        return output
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class ConcatFusion(nn.Module):
    """Simple concatenation-based fusion with MLP processing."""
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super(ConcatFusion, self).__init__()
        
        # Build MLP for processing concatenated features
        input_dim = text_dim + graph_dim
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
        
        self.fusion_mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        text_features: torch.Tensor,
        graph_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through concatenation fusion."""
        # Concatenate features
        concatenated = torch.cat([text_features, graph_features], dim=-1)
        
        # Process through MLP
        fused = self.fusion_mlp(concatenated)
        
        return fused


class AdditiveFusion(nn.Module):
    """Additive fusion with learnable projection and gating."""
    
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super(AdditiveFusion, self).__init__()
        
        # Project both modalities to common dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output processing
        layers = []
        current_dim = hidden_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.output_mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        text_features: torch.Tensor,
        graph_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through additive fusion."""
        # Project to common dimension
        text_proj = self.text_proj(text_features)
        graph_proj = self.graph_proj(graph_features)
        
        # Compute gating weights
        gate_input = torch.cat([text_proj, graph_proj], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # Weighted addition
        fused = gate_weights * text_proj + (1 - gate_weights) * graph_proj
        
        # Process through output MLP
        output = self.output_mlp(fused)
        
        return output


class EarlyExitModule(nn.Module):
    """
    Early-exit inference mechanism with entropy-based dynamic stopping.
    
    This module implements the early-exit strategy described in ENGINE, where
    nodes are processed layer-by-layer until the entropy change drops below
    a threshold, enabling adaptive computation.
    
    Features:
    - Entropy-based confidence estimation
    - Dynamic per-node exit decisions
    - Layer-wise prediction heads
    - Adaptive thresholding
    - Computation budget tracking
    
    Args:
        hidden_dims (List[int]): Hidden dimensions for each layer
        num_classes (int): Number of output classes
        entropy_threshold (float): Threshold for entropy change (τ)
        min_layers (int): Minimum number of layers to compute
        max_layers (int): Maximum number of layers to compute
        confidence_aggregation (str): Method for aggregating confidences
    """
    
    def __init__(
        self,
        hidden_dims: List[int],
        num_classes: int,
        entropy_threshold: float = 0.01,
        min_layers: int = 2,
        max_layers: Optional[int] = None,
        confidence_aggregation: str = 'mean'
    ):
        super(EarlyExitModule, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold
        self.min_layers = min_layers
        self.max_layers = max_layers or len(hidden_dims)
        self.confidence_aggregation = confidence_aggregation
        
        # Create prediction heads for each layer
        self.prediction_heads = nn.ModuleList([
            nn.Linear(dim, num_classes) for dim in hidden_dims
        ])
        
        # Track exit statistics
        self.register_buffer('exit_counts', torch.zeros(len(hidden_dims)))
        self.register_buffer('total_inferences', torch.zeros(1))
        
        logger.info(f"EarlyExit initialized: τ={entropy_threshold}, layers={min_layers}-{self.max_layers}")
    
    def forward(
        self,
        layer_representations: List[torch.Tensor],
        training: bool = True,
        return_all_predictions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass with early-exit decisions.
        
        Args:
            layer_representations (List[torch.Tensor]): Representations from each layer
            training (bool): Whether in training mode
            return_all_predictions (bool): Whether to return predictions from all layers
            
        Returns:
            torch.Tensor or Tuple: Final predictions and optionally intermediate results
        """
        batch_size = layer_representations[0].size(0)
        device = layer_representations[0].device
        
        # Initialize tracking variables
        final_predictions = torch.zeros(batch_size, self.num_classes, device=device)
        exit_layer = torch.full((batch_size,), self.max_layers - 1, device=device, dtype=torch.long)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        all_predictions = []
        entropies = []
        
        # Process each layer
        for layer_idx in range(min(len(layer_representations), self.max_layers)):
            if not active_mask.any():
                break  # All samples have exited
            
            # Get current layer representation
            current_repr = layer_representations[layer_idx]
            
            # Compute predictions for current layer
            logits = self.prediction_heads[layer_idx](current_repr)
            probs = F.softmax(logits, dim=-1)
            all_predictions.append(logits)
            
            # Compute entropy for confidence estimation
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            entropies.append(entropy)
            
            # Early exit decision (only after minimum layers)
            if layer_idx >= self.min_layers - 1:
                if layer_idx > 0:
                    # Compute entropy change
                    prev_entropy = entropies[layer_idx - 1]
                    entropy_change = torch.abs(entropy - prev_entropy)
                    
                    # Exit condition: entropy change below threshold
                    exit_condition = entropy_change <= self.entropy_threshold
                    
                    # Apply exit condition only to active samples
                    should_exit = active_mask & exit_condition
                else:
                    # First layer after minimum - use absolute entropy threshold
                    should_exit = active_mask & (entropy <= self.entropy_threshold)
                
                # Update final predictions for exiting samples
                if should_exit.any():
                    final_predictions[should_exit] = logits[should_exit]
                    exit_layer[should_exit] = layer_idx
                    active_mask[should_exit] = False
                    
                    # Update exit statistics
                    if not training:
                        self.exit_counts[layer_idx] += should_exit.sum().item()
        
        # Handle remaining active samples (use last layer)
        if active_mask.any():
            final_predictions[active_mask] = all_predictions[-1][active_mask]
            exit_layer[active_mask] = len(all_predictions) - 1
            
            if not training:
                self.exit_counts[-1] += active_mask.sum().item()
        
        # Update total inference count
        if not training:
            self.total_inferences += batch_size
        
        if return_all_predictions:
            extra_info = {
                'all_predictions': all_predictions,
                'entropies': entropies,
                'exit_layers': exit_layer,
                'entropy_changes': self._compute_entropy_changes(entropies)
            }
            return final_predictions, extra_info
        else:
            return final_predictions
    
    def _compute_entropy_changes(self, entropies: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute entropy changes between consecutive layers."""
        if len(entropies) <= 1:
            return []
        
        changes = []
        for i in range(1, len(entropies)):
            change = torch.abs(entropies[i] - entropies[i-1])
            changes.append(change)
        
        return changes
    
    def get_exit_statistics(self) -> Dict[str, Union[float, List[float]]]:
        """Get early exit statistics."""
        total = self.total_inferences.item()
        
        if total == 0:
            return {
                'total_inferences': 0,
                'average_exit_layer': 0.0,
                'exit_distribution': [0.0] * len(self.hidden_dims),
                'computation_savings': 0.0
            }
        
        # Compute statistics
        exit_dist = (self.exit_counts / total).tolist()
        avg_exit_layer = sum(i * prob for i, prob in enumerate(exit_dist))
        
        # Computation savings compared to always using all layers
        max_layers = len(self.hidden_dims)
        savings = (max_layers - avg_exit_layer) / max_layers
        
        return {
            'total_inferences': int(total),
            'average_exit_layer': avg_exit_layer,
            'exit_distribution': exit_dist,
            'computation_savings': savings
        }
    
    def reset_statistics(self):
        """Reset exit statistics."""
        self.exit_counts.zero_()
        self.total_inferences.zero_()
    
    def set_threshold(self, new_threshold: float):
        """Update entropy threshold for exit decisions."""
        self.entropy_threshold = new_threshold
        logger.info(f"Updated entropy threshold to {new_threshold}")


class AdaptiveComputationModule(nn.Module):
    """
    Adaptive computation module that dynamically adjusts computation based on
    input complexity and available computational budget.
    
    This module extends the basic early-exit mechanism with budget awareness
    and adaptive threshold adjustment based on current computational load.
    """
    
    def __init__(
        self,
        early_exit_module: EarlyExitModule,
        initial_budget: float = 1.0,
        budget_decay: float = 0.95,
        adaptive_threshold: bool = True
    ):
        super(AdaptiveComputationModule, self).__init__()
        
        self.early_exit = early_exit_module
        self.initial_budget = initial_budget
        self.budget_decay = budget_decay
        self.adaptive_threshold = adaptive_threshold
        
        # Budget tracking
        self.register_buffer('current_budget', torch.tensor(initial_budget))
        self.register_buffer('base_threshold', torch.tensor(early_exit_module.entropy_threshold))
    
    def forward(
        self,
        layer_representations: List[torch.Tensor],
        training: bool = True,
        return_all_predictions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass with budget-aware adaptive computation."""
        
        # Adjust threshold based on current budget
        if self.adaptive_threshold and not training:
            # Lower budget -> higher threshold (more aggressive early exit)
            budget_factor = 1.0 / torch.clamp(self.current_budget, min=0.1, max=1.0)
            adjusted_threshold = self.base_threshold * budget_factor
            self.early_exit.entropy_threshold = adjusted_threshold.item()
        
        # Run early exit inference
        results = self.early_exit(
            layer_representations, training, return_all_predictions
        )
        
        # Update budget (decay over time to simulate resource consumption)
        if not training:
            self.current_budget *= self.budget_decay
            
            # Reset budget periodically (simulating batch processing)
            if torch.rand(1).item() < 0.01:  # 1% chance per inference
                self.current_budget = torch.tensor(self.initial_budget)
        
        return results
    
    def get_budget_info(self) -> Dict[str, float]:
        """Get current budget information."""
        return {
            'current_budget': self.current_budget.item(),
            'initial_budget': self.initial_budget,
            'budget_utilization': 1.0 - self.current_budget.item(),
            'current_threshold': self.early_exit.entropy_threshold
        } 