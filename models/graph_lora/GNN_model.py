"""
Graph Neural Network Models with Low-Rank Adaptation (LoRA) Support

This module implements various Graph Neural Network architectures with support for
Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. LoRA enables
efficient adaptation of pre-trained models by learning low-rank decomposition
of weight updates rather than updating full weight matrices.

Key Features:
- Standard GNN architectures (GCN, GAT, TransformerConv)
- LoRA-enhanced versions for parameter-efficient fine-tuning
- Flexible layer configuration and activation functions
- Memory-efficient training and inference

Supported Architectures:
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network  
- TransformerConv: Graph Transformer Convolution

Author: Graph LoRA Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn.dense.linear import Linear
from typing import Tuple, Union, Optional
from torch import Tensor
import math


class GNN(torch.nn.Module):
    """
    Generic Graph Neural Network with configurable architecture.
    
    This class provides a unified interface for different GNN architectures
    (GCN, GAT, TransformerConv) with flexible layer configuration. It supports
    variable number of layers with automatic dimension handling.
    
    Args:
        input_dim (int): Input feature dimension
        out_dim (int): Output embedding dimension
        activation (callable): Activation function to use between layers
        gnn_type (str): Type of GNN layer ('GCN', 'GAT', 'TransformerConv')
        gnn_layer_num (int): Number of GNN layers (minimum 1)
        
    Attributes:
        gnn_layer_num (int): Number of GNN layers
        activation (callable): Activation function
        gnn_type (str): Type of GNN architecture
        conv (nn.ModuleList): List of GNN convolution layers
        
    Raises:
        KeyError: If gnn_type is not supported
        ValueError: If gnn_layer_num is less than 1
    """
    
    def __init__(self, 
                 input_dim: int, 
                 out_dim: int, 
                 activation: callable, 
                 gnn_type: str = 'TransformerConv', 
                 gnn_layer_num: int = 2):
        super().__init__()
        
        # Validate inputs
        if gnn_layer_num < 1:
            raise ValueError(f'GNN layer_num should be >=1 but got {gnn_layer_num}')
        
        # Store configuration
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        self.gnn_type = gnn_type
        
        # Map string to convolution class
        conv_mapping = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'TransformerConv': TransformerConv
        }
        
        if gnn_type not in conv_mapping:
            raise KeyError(f'Unsupported gnn_type: {gnn_type}. '
                          f'Supported types: {list(conv_mapping.keys())}')
        
        GraphConv = conv_mapping[gnn_type]
        
        # Build layer architecture
        self.conv = self._build_layers(GraphConv, input_dim, out_dim, gnn_layer_num)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_layers(self, 
                     GraphConv: torch.nn.Module, 
                     input_dim: int, 
                     out_dim: int, 
                     num_layers: int) -> nn.ModuleList:
        """
        Build the sequence of GNN layers with appropriate dimensions.
        
        Args:
            GraphConv: GNN convolution class
            input_dim (int): Input feature dimension
            out_dim (int): Output feature dimension
            num_layers (int): Number of layers
            
        Returns:
            nn.ModuleList: List of configured GNN layers
        """
        layers = []
        
        if num_layers == 1:
            # Single layer: input_dim -> out_dim
            layers.append(GraphConv(input_dim, out_dim))
        elif num_layers == 2:
            # Two layers: input_dim -> 2*out_dim -> out_dim
            layers.extend([
                GraphConv(input_dim, 2 * out_dim),
                GraphConv(2 * out_dim, out_dim)
            ])
        else:
            # Multiple layers: input_dim -> 2*out_dim -> ... -> 2*out_dim -> out_dim
            layers.append(GraphConv(input_dim, 2 * out_dim))
            
            # Intermediate layers (all have dimension 2*out_dim)
            for _ in range(num_layers - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim))
            
            # Final layer
            layers.append(GraphConv(2 * out_dim, out_dim))
        
        return nn.ModuleList(layers)
    
    def _init_weights(self, module: torch.nn.Module) -> None:
        """
        Initialize weights of linear layers using Xavier uniform initialization.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, (nn.Linear, Linear)):
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, out_dim]
        """
        # Process through all layers except the last
        for conv in self.conv[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
        
        # Final layer without activation
        node_emb = self.conv[-1](x, edge_index)
        return node_emb

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings in evaluation mode.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge connectivity
            
        Returns:
            torch.Tensor: Node embeddings
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index)

    def __repr__(self) -> str:
        """String representation of the GNN model."""
        return (f"GNN(type={self.gnn_type}, layers={self.gnn_layer_num}, "
                f"params={sum(p.numel() for p in self.parameters()):,})")


class GATConv_LoRA(GATConv):
    """
    Graph Attention Convolution with Low-Rank Adaptation (LoRA).
    
    This class extends the standard GAT convolution with LoRA for parameter-efficient
    fine-tuning. Instead of updating the full weight matrices, LoRA learns low-rank
    decompositions of the weight updates: ΔW = BA, where B and A are low-rank matrices.
    
    The LoRA adaptation is applied to the linear transformations in GAT:
    - Source node transformation: W_src + B_src @ A_src  
    - Destination node transformation: W_dst + B_dst @ A_dst
    
    Args:
        in_channels (int or tuple): Input feature dimensions
        out_channels (int): Output feature dimensions  
        heads (int): Number of attention heads
        concat (bool): Whether to concatenate attention heads
        negative_slope (float): LeakyReLU negative slope
        dropout (float): Dropout probability
        add_self_loops (bool): Whether to add self-loops
        edge_dim (int, optional): Edge feature dimension
        fill_value (str or float): Fill value for added self-loops
        bias (bool): Whether to use bias
        r (int): LoRA rank (number of low-rank dimensions)
        **kwargs: Additional arguments for parent class
        
    Attributes:
        r (int): LoRA rank
        lin_src (nn.Sequential): LoRA-enhanced source transformation
        lin_dst (nn.Sequential): LoRA-enhanced destination transformation
        lin_src_a/lin_src_b: LoRA decomposition matrices for source
        lin_dst_a/lin_dst_b: LoRA decomposition matrices for destination
    """
    
    def __init__(self, 
                 in_channels: Union[int, Tuple[int, int]], 
                 out_channels: int, 
                 heads: int = 1, 
                 concat: bool = True, 
                 negative_slope: float = 0.2, 
                 dropout: float = 0.0, 
                 add_self_loops: bool = True, 
                 edge_dim: Optional[int] = None, 
                 fill_value: Union[float, Tensor, str] = 'mean', 
                 bias: bool = True, 
                 r: int = 32, 
                 **kwargs):
        super().__init__(
            in_channels, out_channels, heads, concat, negative_slope, 
            dropout, add_self_loops, edge_dim, fill_value, bias, **kwargs
        )
        
        # Validate LoRA rank
        if r <= 0:
            raise ValueError("LoRA rank 'r' must be positive")
        if r >= min(in_channels if isinstance(in_channels, int) else min(in_channels), 
                   heads * out_channels):
            print(f"Warning: LoRA rank {r} is large relative to matrix dimensions. "
                  f"Consider using a smaller rank for efficiency.")
        
        self.r = r
        
        # Create LoRA decomposition for source transformation
        if isinstance(in_channels, int):
            # Homogeneous graph case
            self.lin_src_a = Linear(in_channels, self.r, bias=False, weight_initializer='glorot')
            self.lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
            self.lin_dst = self.lin_src  # Shared transformation
        else:
            # Heterogeneous graph case
            self.lin_src_a = Linear(in_channels[0], self.r, bias=False, weight_initializer='glorot')
            self.lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
            
            self.lin_dst_a = Linear(in_channels[1], self.r, bias=False, weight_initializer='glorot')
            self.lin_dst_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = nn.Sequential(self.lin_dst_a, self.lin_dst_b)

        # Initialize LoRA weights
        self.reset_parameters_lora()

    def reset_parameters_lora(self) -> None:
        """
        Initialize LoRA parameters using the standard LoRA initialization strategy.
        
        Following the LoRA paper:
        - Matrix A (first transformation) uses Kaiming/Xavier initialization
        - Matrix B (second transformation) is initialized to zeros
        This ensures the LoRA adaptation starts as identity (no change to original weights)
        """
        # Initialize A matrices (random initialization)
        nn.init.kaiming_normal_(self.lin_src_a.weight, a=math.sqrt(5))
        if hasattr(self, 'lin_dst_a'):
            nn.init.kaiming_normal_(self.lin_dst_a.weight, a=math.sqrt(5))
        
        # Initialize B matrices (zero initialization for identity at start)
        nn.init.zeros_(self.lin_src_b.weight)
        if hasattr(self, 'lin_dst_b'):
            nn.init.zeros_(self.lin_dst_b.weight)

    def get_lora_parameters(self) -> list:
        """
        Get only the LoRA parameters for optimization.
        
        Returns:
            list: List of LoRA parameter tensors
        """
        lora_params = [self.lin_src_a.weight, self.lin_src_b.weight]
        if hasattr(self, 'lin_dst_a'):
            lora_params.extend([self.lin_dst_a.weight, self.lin_dst_b.weight])
        return lora_params

    def freeze_base_parameters(self) -> None:
        """Freeze the original GAT parameters, keeping only LoRA parameters trainable."""
        for name, param in super().named_parameters():
            if 'lin_src' not in name and 'lin_dst' not in name:
                param.requires_grad = False

    def __repr__(self) -> str:
        """String representation of the LoRA GAT layer."""
        return (f"GATConv_LoRA(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, heads={self.heads}, "
                f"lora_rank={self.r})")


class GNNLoRA(torch.nn.Module):
    """
    Graph Neural Network with Low-Rank Adaptation (LoRA) for efficient fine-tuning.
    
    This model combines a pre-trained GNN with LoRA layers to enable parameter-efficient
    adaptation to downstream tasks. The original GNN weights are frozen, and only the
    low-rank adaptation matrices are trained.
    
    Architecture:
    - Base GNN (frozen): Pre-trained graph neural network
    - LoRA layers: Low-rank adaptations added to each layer
    - Final output: Combination of base GNN and LoRA outputs
    
    Args:
        input_dim (int): Input feature dimension
        out_dim (int): Output embedding dimension
        activation (callable): Activation function
        gnn (torch.nn.Module): Pre-trained base GNN model
        gnn_type (str): Type of GNN ('GAT', 'GCN', 'TransformerConv')
        gnn_layer_num (int): Number of GNN layers
        r (int): LoRA rank for adaptation matrices
        
    Attributes:
        gnn: Pre-trained base GNN (frozen)
        conv: LoRA adaptation layers
        gnn_layer_num: Number of layers
        activation: Activation function
        gnn_type: Type of GNN architecture
    """
    
    def __init__(self, 
                 input_dim: int, 
                 out_dim: int, 
                 activation: callable, 
                 gnn: torch.nn.Module, 
                 gnn_type: str = 'GAT', 
                 gnn_layer_num: int = 2, 
                 r: int = 32):
        super().__init__()
        
        # Validate inputs
        if gnn_layer_num < 1:
            raise ValueError(f'GNN layer_num should be >=1 but got {gnn_layer_num}')
        if r <= 0:
            raise ValueError("LoRA rank 'r' must be positive")
        
        # Store configuration
        self.gnn = gnn
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        self.gnn_type = gnn_type
        
        # Freeze base GNN parameters
        for param in self.gnn.parameters():
            param.requires_grad = False
        
        # Map string to LoRA convolution class
        conv_mapping = {
            'GCN': GCNConv,  # Note: LoRA version not implemented for GCN yet
            'GAT': GATConv_LoRA,
            'TransformerConv': TransformerConv,  # Note: LoRA version not implemented yet
        }
        
        if gnn_type not in conv_mapping:
            raise KeyError(f'Unsupported gnn_type: {gnn_type}. '
                          f'Supported types: {list(conv_mapping.keys())}')
        
        GraphConv = conv_mapping[gnn_type]
        
        # Build LoRA adaptation layers
        self.conv = self._build_lora_layers(GraphConv, input_dim, out_dim, gnn_layer_num, r)
        
        print(f"Created GNNLoRA with {sum(p.numel() for p in self.parameters() if p.requires_grad):,} "
              f"trainable parameters (LoRA rank: {r})")

    def _build_lora_layers(self, 
                          GraphConv: torch.nn.Module, 
                          input_dim: int, 
                          out_dim: int, 
                          num_layers: int, 
                          r: int) -> nn.ModuleList:
        """
        Build LoRA adaptation layers matching the base GNN architecture.
        
        Args:
            GraphConv: GNN convolution class
            input_dim (int): Input feature dimension
            out_dim (int): Output feature dimension
            num_layers (int): Number of layers
            r (int): LoRA rank
            
        Returns:
            nn.ModuleList: List of LoRA adaptation layers
        """
        layers = []
        
        if num_layers == 1:
            # Single layer: input_dim -> out_dim
            if GraphConv == GATConv_LoRA:
                layers.append(GraphConv(input_dim, out_dim, r=r))
            else:
                layers.append(GraphConv(input_dim, out_dim))
        elif num_layers == 2:
            # Two layers: input_dim -> 2*out_dim -> out_dim
            if GraphConv == GATConv_LoRA:
                layers.extend([
                    GraphConv(input_dim, 2 * out_dim, r=r),
                    GraphConv(2 * out_dim, out_dim, r=r)
                ])
            else:
                layers.extend([
                    GraphConv(input_dim, 2 * out_dim),
                    GraphConv(2 * out_dim, out_dim)
                ])
        else:
            # Multiple layers: input_dim -> 2*out_dim -> ... -> 2*out_dim -> out_dim
            if GraphConv == GATConv_LoRA:
                layers.append(GraphConv(input_dim, 2 * out_dim, r=r))
                
                # Intermediate layers
                for _ in range(num_layers - 2):
                    layers.append(GraphConv(2 * out_dim, 2 * out_dim, r=r))
                
                # Final layer
                layers.append(GraphConv(2 * out_dim, out_dim, r=r))
            else:
                layers.append(GraphConv(input_dim, 2 * out_dim))
                
                # Intermediate layers  
                for _ in range(num_layers - 2):
                    layers.append(GraphConv(2 * out_dim, 2 * out_dim))
                
                # Final layer
                layers.append(GraphConv(2 * out_dim, out_dim))
        
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass combining base GNN and LoRA adaptations.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Combined embeddings (base + LoRA)
                - Base GNN embeddings only
                - LoRA adaptation embeddings only
        """
        # Store intermediate outputs for combining
        x_base = x.clone()
        x_lora = x.clone()
        
        # Process through each layer
        for i in range(self.gnn_layer_num):
            # Base GNN forward pass (frozen)
            if i < len(self.gnn.conv):
                x_base = self.gnn.conv[i](x_base, edge_index)
                if i < self.gnn_layer_num - 1:  # Apply activation except for last layer
                    x_base = self.activation(x_base)
            
            # LoRA adaptation forward pass
            x_lora = self.conv[i](x_lora, edge_index)
            if i < self.gnn_layer_num - 1:  # Apply activation except for last layer
                x_lora = self.activation(x_lora)
        
        # Combine base and LoRA outputs
        combined_emb = x_base + x_lora
        
        return combined_emb, x_base, x_lora

    def get_lora_parameters(self) -> list:
        """
        Get all LoRA parameters for optimization.
        
        Returns:
            list: List of LoRA parameter tensors
        """
        lora_params = []
        for layer in self.conv:
            if hasattr(layer, 'get_lora_parameters'):
                lora_params.extend(layer.get_lora_parameters())
        return lora_params

    def get_trainable_parameters(self) -> list:
        """
        Get all trainable parameters (should be only LoRA parameters).
        
        Returns:
            list: List of trainable parameter tensors
        """
        return [p for p in self.parameters() if p.requires_grad]

    def save_lora_weights(self, path: str) -> None:
        """
        Save only the LoRA adaptation weights.
        
        Args:
            path (str): Path to save LoRA weights
        """
        lora_state_dict = {name: param for name, param in self.named_parameters() 
                          if param.requires_grad}
        torch.save(lora_state_dict, path)

    def load_lora_weights(self, path: str) -> None:
        """
        Load LoRA adaptation weights.
        
        Args:
            path (str): Path to load LoRA weights from
        """
        lora_state_dict = torch.load(path)
        self.load_state_dict(lora_state_dict, strict=False)

    def __repr__(self) -> str:
        """String representation of the GNNLoRA model."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        efficiency = trainable_params / total_params * 100
        
        return (f"GNNLoRA(type={self.gnn_type}, layers={self.gnn_layer_num}, "
                f"trainable_params={trainable_params:,}/{total_params:,} "
                f"({efficiency:.2f}%))")