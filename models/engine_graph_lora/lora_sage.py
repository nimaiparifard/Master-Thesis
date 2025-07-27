"""
LoRA-Enhanced Graph Neural Networks

This module implements GraphSAGE and GAT architectures enhanced with Low-Rank Adaptation
(LoRA) for parameter-efficient fine-tuning. The LoRA technique allows efficient adaptation
of pre-trained GNNs to new domains with minimal parameter overhead.

Key Features:
- LoRA-enhanced GraphSAGE and GAT implementations
- Parameter-efficient adaptation with rank decomposition
- Support for different aggregation strategies
- Memory-efficient training and inference
- Configurable rank and scaling parameters

Mathematical Foundation:
W' = W + A·B, where A∈ℝ^(d×r), B∈ℝ^(r×d) (r≪d)

Author: Graph Neural Network Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GINConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Union, Tuple, List
import math


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for parameter-efficient fine-tuning.
    
    This layer implements the LoRA technique by decomposing weight updates into
    low-rank matrices, significantly reducing the number of trainable parameters.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension  
        rank (int): Rank of the low-rank decomposition
        alpha (float): Scaling parameter for LoRA weights
        dropout (float): Dropout rate for regularization
        merge_weights (bool): Whether to merge LoRA weights with base weights
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        super(LoRALayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        
        # LoRA matrices: W = W_base + α/r * A @ B
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Base weight placeholder (will be set externally)
        self.base_weight = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with random values, B with zeros (as in original paper)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x (torch.Tensor): Input tensor [*, in_features]
            
        Returns:
            torch.Tensor: Output tensor [*, out_features]
        """
        # Base transformation (if base weight is provided)
        if self.base_weight is not None:
            base_output = F.linear(x, self.base_weight)
        else:
            base_output = 0
        
        # LoRA transformation: scaling * B(A(x))
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
        return base_output + lora_output
    
    def set_base_weight(self, weight: torch.Tensor):
        """Set the base weight matrix."""
        self.base_weight = weight.detach()
        if hasattr(self.base_weight, 'requires_grad'):
            self.base_weight.requires_grad = False


class LoRASAGEConv(MessagePassing):
    """
    GraphSAGE convolution with LoRA adaptation.
    
    This layer combines the GraphSAGE architecture with LoRA for efficient adaptation.
    It supports various aggregation functions and normalization strategies.
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        rank (int): LoRA rank
        alpha (float): LoRA scaling parameter
        aggr (str): Aggregation method ('mean', 'max', 'lstm')
        normalize (bool): Whether to apply L2 normalization
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 8,
        alpha: float = 32.0,
        aggr: str = 'mean',
        normalize: bool = False,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super(LoRASAGEConv, self).__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        # Base SAGE layer (frozen)
        self.base_sage = SAGEConv(in_channels, out_channels, normalize=normalize, bias=bias)
        
        # Freeze base parameters
        for param in self.base_sage.parameters():
            param.requires_grad = False
        
        # LoRA adaptations for lin_l (self transformation) and lin_r (neighbor transformation)
        self.lora_lin_l = LoRALayer(in_channels, out_channels, rank, alpha, dropout)
        self.lora_lin_r = LoRALayer(in_channels, out_channels, rank, alpha, dropout)
        
        # Set base weights
        self.lora_lin_l.set_base_weight(self.base_sage.lin_l.weight)
        self.lora_lin_r.set_base_weight(self.base_sage.lin_r.weight)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Forward pass of LoRA-enhanced SAGE convolution.
        
        Args:
            x (torch.Tensor): Input node features [num_nodes, in_channels]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            size (Optional[Tuple[int, int]]): Size of source and target nodes
            
        Returns:
            torch.Tensor: Output node features [num_nodes, out_channels]
        """
        # Self-transformation with LoRA
        x_self = self.lora_lin_l(x)
        
        # Message passing for neighbor aggregation
        x_neighbors = self.propagate(edge_index, x=x, size=size)
        
        # Combine self and neighbor representations
        out = x_self + x_neighbors
        
        # Apply normalization if specified
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Message function for neighbor aggregation."""
        return self.lora_lin_r(x_j)


class LoRAGATConv(MessagePassing):
    """
    Graph Attention Network (GAT) convolution with LoRA adaptation.
    
    This layer implements GAT with LoRA modifications for efficient fine-tuning.
    It includes multi-head attention and supports various attention mechanisms.
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension per head
        heads (int): Number of attention heads
        rank (int): LoRA rank
        alpha (float): LoRA scaling parameter
        dropout (float): Dropout rate for attention coefficients
        edge_dim (Optional[int]): Edge feature dimension
        concat (bool): Whether to concatenate or average multi-head outputs
        negative_slope (float): LeakyReLU negative slope
        bias (bool): Whether to use bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True
    ):
        super(LoRAGATConv, self).__init__(node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Base GAT layer (frozen)
        self.base_gat = GATConv(
            in_channels, out_channels, heads=heads, concat=concat,
            dropout=dropout, edge_dim=edge_dim, negative_slope=negative_slope, bias=bias
        )
        
        # Freeze base parameters
        for param in self.base_gat.parameters():
            param.requires_grad = False
        
        # LoRA adaptations for linear transformations
        self.lora_lin = LoRALayer(
            in_channels, heads * out_channels, rank, alpha, dropout
        )
        
        # LoRA for attention mechanism
        self.lora_att_src = LoRALayer(out_channels, 1, rank//2, alpha, dropout)
        self.lora_att_dst = LoRALayer(out_channels, 1, rank//2, alpha, dropout)
        
        if edge_dim is not None:
            self.lora_lin_edge = LoRALayer(edge_dim, heads * out_channels, rank, alpha, dropout)
        else:
            self.lora_lin_edge = None
        
        # Set base weights
        self.lora_lin.set_base_weight(self.base_gat.lin.weight)
        self.lora_att_src.set_base_weight(self.base_gat.att_src.view(-1, 1))
        self.lora_att_dst.set_base_weight(self.base_gat.att_dst.view(-1, 1))
        
        if hasattr(self.base_gat, 'lin_edge') and self.lora_lin_edge is not None:
            self.lora_lin_edge.set_base_weight(self.base_gat.lin_edge.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of LoRA-enhanced GAT convolution.
        
        Args:
            x (torch.Tensor): Input node features [num_nodes, in_channels]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_attr (Optional[torch.Tensor]): Edge features [num_edges, edge_dim]
            return_attention_weights (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor or Tuple: Output features and optionally attention weights
        """
        H, C = self.heads, self.out_channels
        num_nodes = x.size(0)
        
        # Linear transformation with LoRA
        x = self.lora_lin(x).view(-1, H, C)
        
        # Add self-loops
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, num_nodes=num_nodes
        )
        
        # Propagate messages
        out, attention = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, 
            return_attention_weights=return_attention_weights
        )
        
        # Concatenate or average multi-head outputs
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        
        if return_attention_weights:
            return out, attention
        else:
            return out
    
    def message(
        self, 
        x_i: torch.Tensor, 
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        index: torch.Tensor = None,
        ptr: Optional[torch.Tensor] = None,
        size_i: Optional[int] = None
    ) -> torch.Tensor:
        """Compute messages with LoRA-enhanced attention."""
        # Compute attention coefficients with LoRA
        alpha_src = self.lora_att_src(x_i).squeeze(-1)  # [num_edges, heads]
        alpha_dst = self.lora_att_dst(x_j).squeeze(-1)  # [num_edges, heads]
        
        alpha = alpha_src + alpha_dst
        
        # Add edge features if available
        if edge_attr is not None and self.lora_lin_edge is not None:
            edge_attr = self.lora_lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (edge_attr * x_j).sum(dim=-1)
        
        # Apply LeakyReLU and softmax
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self.softmax(alpha, index, ptr, size_i)
        
        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to node features
        return x_j * alpha.unsqueeze(-1)


class LoRASAGE(nn.Module):
    """
    Multi-layer LoRA-enhanced GraphSAGE model.
    
    This model stacks multiple LoRA-SAGE layers with normalization and activation.
    It provides a complete GNN architecture for node representation learning.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dims (List[int]): Hidden layer dimensions
        output_dim (int): Output feature dimension
        rank (int): LoRA rank for all layers
        alpha (float): LoRA scaling parameter
        num_layers (int): Number of SAGE layers
        dropout (float): Dropout rate
        activation (str): Activation function ('relu', 'gelu', 'tanh')
        normalize (bool): Whether to apply layer normalization
        residual (bool): Whether to use residual connections
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        rank: int = 8,
        alpha: float = 32.0,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        normalize: bool = True,
        residual: bool = True
    ):
        super(LoRASAGE, self).__init__()
        
        self.num_layers = num_layers
        self.residual = residual
        
        # Build layer dimensions
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * (num_layers - 1)
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # LoRA-SAGE layers
        self.convs = nn.ModuleList([
            LoRASAGEConv(dims[i], dims[i+1], rank=rank, alpha=alpha, dropout=dropout)
            for i in range(num_layers)
        ])
        
        # Normalization layers
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i+1]) if normalize else nn.Identity()
            for i in range(num_layers)
        ])
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection layers (for dimension matching)
        self.residual_projs = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) if dims[i] != dims[i+1] and residual else nn.Identity()
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA-SAGE layers.
        
        Args:
            x (torch.Tensor): Input node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            
        Returns:
            torch.Tensor: Output node representations [num_nodes, output_dim]
        """
        for i, (conv, norm, residual_proj) in enumerate(zip(self.convs, self.norms, self.residual_projs)):
            x_input = x
            
            # Graph convolution
            x = conv(x, edge_index)
            
            # Normalization
            x = norm(x)
            
            # Residual connection (if applicable)
            if self.residual and i < len(self.convs) - 1:  # No residual on last layer
                x = x + residual_proj(x_input)
            
            # Activation (except last layer)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        return x
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRAGAT(nn.Module):
    """
    Multi-layer LoRA-enhanced Graph Attention Network.
    
    Similar to LoRASAGE but uses attention mechanisms for message aggregation.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dims (List[int]): Hidden layer dimensions  
        output_dim (int): Output feature dimension
        heads (int): Number of attention heads
        rank (int): LoRA rank
        alpha (float): LoRA scaling parameter
        num_layers (int): Number of GAT layers
        dropout (float): Dropout rate
        concat_heads (bool): Whether to concatenate attention heads
        residual (bool): Whether to use residual connections
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        heads: int = 8,
        rank: int = 8,
        alpha: float = 32.0,
        num_layers: int = 2,
        dropout: float = 0.1,
        concat_heads: bool = True,
        residual: bool = True
    ):
        super(LoRAGAT, self).__init__()
        
        self.num_layers = num_layers
        self.residual = residual
        
        # Build layer dimensions (accounting for attention head concatenation)
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * (num_layers - 1)
        
        # Adjust dimensions for multi-head attention
        effective_hidden_dims = []
        for i, dim in enumerate(hidden_dims):
            if i < num_layers - 1 and concat_heads:
                effective_hidden_dims.append(dim // heads)
            else:
                effective_hidden_dims.append(dim)
        
        dims = [input_dim] + effective_hidden_dims + [output_dim]
        
        # LoRA-GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            concat = concat_heads if i < num_layers - 1 else False
            self.convs.append(
                LoRAGATConv(
                    dims[i], dims[i+1], heads=heads, rank=rank, 
                    alpha=alpha, dropout=dropout, concat=concat
                )
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through LoRA-GAT layers.
        
        Args:
            x (torch.Tensor): Input node features
            edge_index (torch.Tensor): Edge indices
            return_attention_weights (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor or Tuple: Output features and optionally attention weights
        """
        attention_weights = [] if return_attention_weights else None
        
        for i, conv in enumerate(self.convs):
            if return_attention_weights:
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                x = conv(x, edge_index)
            
            # Apply dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        if return_attention_weights:
            return x, attention_weights
        else:
            return x
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 