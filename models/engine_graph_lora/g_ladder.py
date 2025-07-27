"""
G-Ladder Modules: ENGINE-style Cross-Layer Graph-Text Fusion

This module implements G-Ladder components that inject graph context into 
transformer layers for cross-modal learning between text and graph structure.

Based on the ENGINE paper architecture with LoRA adaptations for efficiency.

Author: Graph Neural Network Research Team
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer for parameter-efficient fine-tuning
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen base layer
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # LoRA adaptation matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base transformation (frozen)
        base_output = self.base_layer(x)
        
        # LoRA adaptation
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        
        return base_output + self.scaling * lora_output


class GraphMessagePassing(MessagePassing):
    """
    Efficient message passing module for G-Ladder
    """
    
    def __init__(
        self,
        hidden_size: int,
        mp_type: str = "mean",
        add_self_loops: bool = True,
        normalize: bool = True
    ):
        super().__init__(aggr=mp_type)
        self.hidden_size = hidden_size
        self.mp_type = mp_type
        self.add_self_loops_flag = add_self_loops
        self.normalize = normalize
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, hidden_size]
            edge_index: Edge indices [2, num_edges]
        """
        if self.add_self_loops_flag:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            
        # Compute normalization
        if self.normalize:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = None
            
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j: torch.Tensor, norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        if norm is not None:
            return norm.view(-1, 1) * x_j
        return x_j


class GLadderModule(nn.Module):
    """
    Single G-Ladder module that injects graph context into transformer layer
    
    Implements: h̃^(ℓ) = h^(ℓ) + α * MP(h^(ℓ), A)
    where MP is differentiable message passing and α is learnable gating
    """
    
    def __init__(
        self,
        hidden_size: int,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        mp_type: str = "mean",
        use_gating: bool = True,
        gate_activation: str = "sigmoid"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_gating = use_gating
        
        # Message passing component
        self.message_passing = GraphMessagePassing(
            hidden_size=hidden_size,
            mp_type=mp_type,
            add_self_loops=True,
            normalize=True
        )
        
        # LoRA-enhanced message transformation
        self.message_transform = LoRALinear(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
        
        # Gating mechanism
        if use_gating:
            self.gate_transform = LoRALinear(
                in_features=hidden_size,
                out_features=hidden_size,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            
            if gate_activation == "sigmoid":
                self.gate_activation = torch.sigmoid
            elif gate_activation == "tanh":
                self.gate_activation = torch.tanh
            else:
                self.gate_activation = torch.relu
        else:
            # Learnable scalar gating parameter
            self.alpha = nn.Parameter(torch.ones(1))
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input hidden states [num_nodes, hidden_size]
            edge_index: Graph edge indices [2, num_edges]
            num_nodes: Number of nodes (inferred if None)
            
        Returns:
            Enhanced hidden states with graph context
        """
        if num_nodes is None:
            num_nodes = hidden_states.size(0)
            
        # Ensure edge_index has correct node indices
        if edge_index.numel() > 0:
            max_node_idx = edge_index.max().item()
            if max_node_idx >= num_nodes:
                # Filter edges to valid node range
                valid_edges = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_edges]
        
        # Message passing aggregation
        if edge_index.numel() > 0:
            message_out = self.message_passing(hidden_states, edge_index)
        else:
            # No edges case
            message_out = torch.zeros_like(hidden_states)
            
        # Transform messages
        message_transformed = self.message_transform(message_out)
        
        # Apply gating
        if self.use_gating:
            gate_values = self.gate_activation(self.gate_transform(hidden_states))
            gated_messages = gate_values * message_transformed
        else:
            gated_messages = self.alpha * message_transformed
            
        # Residual connection + layer norm
        output = self.layer_norm(hidden_states + gated_messages)
        
        return output


class GLadderLM(nn.Module):
    """
    Complete G-Ladder system that integrates with language model layers
    
    Injects graph context at specified transformer layers using G-Ladder modules
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        mp_type: str = "mean",
        use_gating: bool = True,
        gate_activation: str = "sigmoid",
        injection_layers: Optional[List[int]] = None,
        weight_sharing: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.injection_layers = injection_layers or list(range(0, num_layers, 2))
        self.weight_sharing = weight_sharing
        
        # Create G-Ladder modules
        if weight_sharing:
            # Single shared module
            self.shared_module = GLadderModule(
                hidden_size=hidden_size,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                mp_type=mp_type,
                use_gating=use_gating,
                gate_activation=gate_activation
            )
            self.modules_dict = {}
        else:
            # Separate module for each injection layer
            self.modules_dict = nn.ModuleDict()
            for layer_idx in self.injection_layers:
                self.modules_dict[str(layer_idx)] = GLadderModule(
                    hidden_size=hidden_size,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    mp_type=mp_type,
                    use_gating=use_gating,
                    gate_activation=gate_activation
                )
                
    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply G-Ladder fusion for specific layer or all layers
        
        Args:
            hidden_states: Input hidden states [num_nodes, hidden_size]
            edge_index: Graph edge indices [2, num_edges]
            num_nodes: Number of nodes
            layer_idx: Specific layer index (if None, processes all injection layers)
            
        Returns:
            Enhanced hidden states
        """
        if layer_idx is not None:
            # Single layer processing
            if layer_idx in self.injection_layers:
                return self._apply_single_layer(hidden_states, edge_index, num_nodes, layer_idx)
            else:
                return hidden_states
        else:
            # Multi-layer processing (simulated)
            current_states = hidden_states
            for l_idx in self.injection_layers:
                current_states = self._apply_single_layer(current_states, edge_index, num_nodes, l_idx)
            return current_states
    
    def _apply_single_layer(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int],
        layer_idx: int
    ) -> torch.Tensor:
        """Apply G-Ladder module for a specific layer"""
        if self.weight_sharing:
            module = self.shared_module
        else:
            module = self.modules_dict[str(layer_idx)]
            
        return module(hidden_states, edge_index, num_nodes)
    
    def get_injection_layers(self) -> List[int]:
        """Get list of layers where G-Ladder modules are injected"""
        return self.injection_layers
    
    def set_injection_layers(self, layers: List[int]):
        """Update injection layers (useful for dynamic configuration)"""
        self.injection_layers = layers
        
        if not self.weight_sharing:
            # Update modules dictionary
            new_modules = nn.ModuleDict()
            for layer_idx in layers:
                if str(layer_idx) in self.modules_dict:
                    new_modules[str(layer_idx)] = self.modules_dict[str(layer_idx)]
                else:
                    # Create new module for new layer
                    new_modules[str(layer_idx)] = GLadderModule(
                        hidden_size=self.hidden_size,
                        lora_rank=16,  # Default values
                        lora_alpha=32.0,
                        lora_dropout=0.1,
                        mp_type="mean",
                        use_gating=True,
                        gate_activation="sigmoid"
                    )
            self.modules_dict = new_modules


class HookedGLadderLM(GLadderLM):
    """
    G-Ladder system with forward hooks for seamless LLM integration
    
    This version can be attached to pre-trained language models using hooks
    to inject graph context without modifying the original model architecture
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = []
        self.edge_index = None
        self.num_nodes = None
        
    def register_hooks(self, llm_model):
        """
        Register forward hooks on LLM layers
        
        Args:
            llm_model: Pre-trained language model
        """
        self.remove_hooks()  # Remove existing hooks
        
        # Get transformer layers
        if hasattr(llm_model, 'transformer'):
            layers = llm_model.transformer.h  # GPT-style
        elif hasattr(llm_model, 'encoder'):
            layers = llm_model.encoder.layer  # BERT-style
        else:
            raise ValueError("Unsupported LLM architecture")
        
        # Register hooks on injection layers
        for layer_idx in self.injection_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(
                    self._make_hook_fn(layer_idx)
                )
                self.hooks.append(hook)
    
    def _make_hook_fn(self, layer_idx: int):
        """Create hook function for specific layer"""
        def hook_fn(module, input, output):
            if self.edge_index is not None and self.num_nodes is not None:
                # Extract hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Apply G-Ladder fusion
                enhanced_states = self._apply_single_layer(
                    hidden_states.squeeze(0),  # Remove batch dim if present
                    self.edge_index,
                    self.num_nodes,
                    layer_idx
                )
                
                # Return modified output
                if isinstance(output, tuple):
                    return (enhanced_states.unsqueeze(0),) + output[1:]
                else:
                    return enhanced_states.unsqueeze(0)
            return output
        
        return hook_fn
    
    def set_graph_context(self, edge_index: torch.Tensor, num_nodes: int):
        """Set graph context for hook functions"""
        self.edge_index = edge_index
        self.num_nodes = num_nodes
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        """Cleanup hooks on deletion"""
        self.remove_hooks()


# Utility functions for G-Ladder integration

def create_g_ladder_config(
    hidden_size: int,
    num_layers: int,
    injection_strategy: str = "uniform",
    injection_density: float = 0.5
) -> Dict[str, Any]:
    """
    Create G-Ladder configuration based on strategy
    
    Args:
        hidden_size: Hidden dimension size
        num_layers: Total number of transformer layers
        injection_strategy: Strategy for layer selection ("uniform", "early", "late", "custom")
        injection_density: Fraction of layers to inject (for uniform strategy)
        
    Returns:
        Configuration dictionary
    """
    if injection_strategy == "uniform":
        step = max(1, int(1 / injection_density))
        injection_layers = list(range(0, num_layers, step))
    elif injection_strategy == "early":
        num_inject = max(1, int(num_layers * injection_density))
        injection_layers = list(range(num_inject))
    elif injection_strategy == "late":
        num_inject = max(1, int(num_layers * injection_density))
        injection_layers = list(range(num_layers - num_inject, num_layers))
    else:
        # Default to every other layer
        injection_layers = list(range(0, num_layers, 2))
    
    return {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'injection_layers': injection_layers,
        'lora_rank': min(64, hidden_size // 16),
        'lora_alpha': 32.0,
        'lora_dropout': 0.1,
        'mp_type': 'mean',
        'use_gating': True,
        'gate_activation': 'sigmoid',
        'weight_sharing': False
    }


def estimate_g_ladder_parameters(config: Dict[str, Any]) -> int:
    """
    Estimate number of trainable parameters for G-Ladder modules
    
    Args:
        config: G-Ladder configuration
        
    Returns:
        Estimated parameter count
    """
    hidden_size = config['hidden_size']
    lora_rank = config['lora_rank']
    injection_layers = config['injection_layers']
    weight_sharing = config['weight_sharing']
    use_gating = config['use_gating']
    
    # Parameters per module
    params_per_module = 0
    
    # Message transform LoRA: 2 * hidden_size * rank
    params_per_module += 2 * hidden_size * lora_rank
    
    # Gate transform LoRA (if used)
    if use_gating:
        params_per_module += 2 * hidden_size * lora_rank
    else:
        params_per_module += 1  # Alpha parameter
    
    # Layer norm: 2 * hidden_size
    params_per_module += 2 * hidden_size
    
    # Total parameters
    if weight_sharing:
        total_params = params_per_module
    else:
        total_params = params_per_module * len(injection_layers)
    
    return total_params 