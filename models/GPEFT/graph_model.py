"""
Advanced Graph Neural Network Architecture Module

This module provides a comprehensive framework for building and managing Graph Neural
Networks (GNNs) with support for multiple architectures, flexible configurations,
and efficient implementations. It includes a unified registration system for models
and encoders, enabling easy extensibility and experimentation.

Key Features:
- Multiple GNN architectures (GCN, SAGE, GAT, GIN, GCNII, MLP)
- Unified model registration and management system
- Flexible encoder-decoder architectures
- Advanced normalization and activation options
- Efficient subgraph and full-graph processing
- Memory-optimized implementations

Performance Optimizations:
- Efficient tensor operations and memory management
- Optimized forward pass implementations
- Support for various normalization techniques
- Flexible activation function selection
- Gradient-friendly architectures

Author: Graph Neural Network Research Team
Version: 3.0 (Completely Rewritten and Optimized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, Dropout, LayerNorm, BatchNorm1d, Identity
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, GCN2Conv, 
    global_mean_pool, global_max_pool, global_add_pool,
    MessagePassing
)
from torch_geometric.nn.inits import glorot, zeros
from typing import Dict, Any, Optional, List, Union, Callable
import warnings
import logging
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Advanced registry system for managing models, encoders, and other components.
    
    This registry provides a centralized system for registering and accessing
    different neural network components, enabling modular design and easy
    experimentation with different architectures.
    
    Features:
    - Centralized component registration
    - Type-safe access to registered components
    - Support for multiple component types
    - Automatic validation and error handling
    """
    
    def __init__(self):
        """Initialize the registry with empty component dictionaries."""
        self.models: Dict[str, type] = {}
        self.encoders: Dict[str, type] = {}
        self.decoders: Dict[str, type] = {}
        self.datasets: Dict[str, type] = {}
        self.samplers: Dict[str, type] = {}
        self.optimizers: Dict[str, type] = {}
        
        logger.info("ModelRegistry initialized")
    
    def register_model(self, model_class: type) -> type:
        """
        Register a model class.
        
        Args:
            model_class (type): Model class to register
            
        Returns:
            type: The registered model class (for decorator usage)
        """
        name = model_class.__name__
        if name in self.models:
            logger.warning(f"Overwriting existing model registration: {name}")
        
        self.models[name] = model_class
        logger.debug(f"Registered model: {name}")
        return model_class
    
    def register_encoder(self, encoder_class: type) -> type:
        """
        Register an encoder class.
        
        Args:
            encoder_class (type): Encoder class to register
            
        Returns:
            type: The registered encoder class (for decorator usage)
        """
        name = encoder_class.__name__
        if name in self.encoders:
            logger.warning(f"Overwriting existing encoder registration: {name}")
        
        self.encoders[name] = encoder_class
        logger.debug(f"Registered encoder: {name}")
        return encoder_class
    
    def register_decoder(self, decoder_class: type) -> type:
        """Register a decoder class."""
        name = decoder_class.__name__
        self.decoders[name] = decoder_class
        logger.debug(f"Registered decoder: {name}")
        return decoder_class
    
    def get_model(self, name: str, **kwargs) -> nn.Module:
        """
        Instantiate a registered model.
        
        Args:
            name (str): Name of the registered model
            **kwargs: Arguments to pass to model constructor
            
        Returns:
            nn.Module: Instantiated model
            
        Raises:
            KeyError: If model name is not registered
        """
        if name not in self.models:
            available = list(self.models.keys())
            raise KeyError(f"Model '{name}' not registered. Available: {available}")
        
        return self.models[name](**kwargs)
    
    def get_encoder(self, name: str, **kwargs) -> nn.Module:
        """
        Instantiate a registered encoder.
        
        Args:
            name (str): Name of the registered encoder
            **kwargs: Arguments to pass to encoder constructor
            
        Returns:
            nn.Module: Instantiated encoder
        """
        if name not in self.encoders:
            available = list(self.encoders.keys())
            raise KeyError(f"Encoder '{name}' not registered. Available: {available}")
        
        return self.encoders[name](**kwargs)
    
    def list_components(self) -> Dict[str, List[str]]:
        """
        List all registered components.
        
        Returns:
            Dict[str, List[str]]: Dictionary with component types and their names
        """
        return {
            'models': list(self.models.keys()),
            'encoders': list(self.encoders.keys()),
            'decoders': list(self.decoders.keys()),
            'datasets': list(self.datasets.keys()),
            'samplers': list(self.samplers.keys()),
            'optimizers': list(self.optimizers.keys())
        }


# Global registry instance
registry = ModelRegistry()


class ActivationFactory:
    """Factory class for creating activation functions."""
    
    @staticmethod
    def get_activation(name: str) -> nn.Module:
        """
        Get activation function by name.
        
        Args:
            name (str): Name of activation function
            
        Returns:
            nn.Module: Activation function module
        """
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'hardtanh': nn.Hardtanh(),
            'prelu': nn.PReLU(),
            'identity': nn.Identity()
        }
        
        if name.lower() not in activations:
            logger.warning(f"Unknown activation '{name}', using ReLU")
            return nn.ReLU()
        
        return activations[name.lower()]


class NormalizationFactory:
    """Factory class for creating normalization layers."""
    
    @staticmethod
    def get_normalization(name: str, num_features: int) -> nn.Module:
        """
        Get normalization layer by name.
        
        Args:
            name (str): Name of normalization layer
            num_features (int): Number of features to normalize
            
        Returns:
            nn.Module: Normalization layer
        """
        normalizations = {
            'batch_norm': BatchNorm1d(num_features),
            'bn': BatchNorm1d(num_features),
            'layer_norm': LayerNorm(num_features),
            'ln': LayerNorm(num_features),
            'identity': Identity(),
            'id': Identity(),
            'none': Identity()
        }
        
        if name.lower() not in normalizations:
            logger.warning(f"Unknown normalization '{name}', using Identity")
            return Identity()
        
        return normalizations[name.lower()]


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for graph encoders.
    
    This class provides a common interface for all graph encoders,
    ensuring consistency and enabling polymorphic usage.
    """
    
    def __init__(
        self,
        input_dim: int,
        layer_num: int = 2,
        hidden_size: int = 128,
        output_dim: int = 128,
        activation: str = "relu",
        dropout: float = 0.5,
        norm: str = 'batch_norm',
        last_activation: bool = True
    ):
        """
        Initialize base encoder.
        
        Args:
            input_dim (int): Input feature dimension
            layer_num (int): Number of layers
            hidden_size (int): Hidden layer dimension
            output_dim (int): Output feature dimension
            activation (str): Activation function name
            dropout (float): Dropout rate
            norm (str): Normalization layer type
            last_activation (bool): Whether to apply activation to last layer
        """
        super(BaseEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.last_activation = last_activation
        
        # Create activation and dropout
        self.activation = ActivationFactory.get_activation(activation)
        self.dropout = Dropout(dropout)
        
        # Store normalization type for creating layers
        self.norm_type = norm
        
        # Initialize layer containers
        self.convs = ModuleList()
        self.norms = ModuleList()
        
        # Build layers
        self._build_layers()
        
        logger.debug(f"Initialized {self.__class__.__name__}: "
                    f"{input_dim} -> {hidden_size} x{layer_num-1} -> {output_dim}")
    
    @abstractmethod
    def _build_layers(self):
        """Build the encoder layers. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass. Must be implemented by subclasses."""
        pass
    
    def reset_parameters(self):
        """Reset parameters of all layers."""
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()


@registry.register_encoder
class GCNEncoder(BaseEncoder):
    """
    Graph Convolutional Network (GCN) encoder.
    
    This encoder implements the Graph Convolutional Network architecture
    with flexible normalization, activation, and dropout options.
    
    Paper: "Semi-Supervised Classification with Graph Convolutional Networks"
    """
    
    def _build_layers(self):
        """Build GCN layers with normalization and activation."""
        if self.layer_num == 1:
            # Single layer case
            self.convs.append(GCNConv(self.input_dim, self.output_dim))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
        else:
            # Multi-layer case
            # First layer
            self.convs.append(GCNConv(self.input_dim, self.hidden_size))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Hidden layers
            for _ in range(self.layer_num - 2):
                self.convs.append(GCNConv(self.hidden_size, self.hidden_size))
                self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Output layer
            self.convs.append(GCNConv(self.hidden_size, self.output_dim))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through GCN layers.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_weight (Optional[torch.Tensor]): Edge weights [num_edges]
            
        Returns:
            torch.Tensor: Output node features [num_nodes, output_dim]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_weight)
            x = norm(x)
            
            # Apply activation (skip for last layer if last_activation=False)
            if i == self.layer_num - 1 and not self.last_activation:
                pass
            else:
                x = self.activation(x)
            
            x = self.dropout(x)
        
        return x


@registry.register_encoder
class SAGEEncoder(BaseEncoder):
    """
    GraphSAGE encoder with sampling-based aggregation.
    
    This encoder implements the GraphSAGE architecture which is particularly
    effective for large graphs and inductive learning scenarios.
    
    Paper: "Inductive Representation Learning on Large Graphs"
    """
    
    def _build_layers(self):
        """Build GraphSAGE layers."""
        if self.layer_num == 1:
            self.convs.append(SAGEConv(self.input_dim, self.output_dim))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
        else:
            # First layer
            self.convs.append(SAGEConv(self.input_dim, self.hidden_size))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Hidden layers
            for _ in range(self.layer_num - 2):
                self.convs.append(SAGEConv(self.hidden_size, self.hidden_size))
                self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Output layer
            self.convs.append(SAGEConv(self.hidden_size, self.output_dim))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through GraphSAGE layers."""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            
            if i == self.layer_num - 1 and not self.last_activation:
                pass
            else:
                x = self.activation(x)
            
            x = self.dropout(x)
        
        return x


@registry.register_encoder
class GATEncoder(BaseEncoder):
    """
    Graph Attention Network (GAT) encoder.
    
    This encoder implements the Graph Attention Network with multi-head
    attention mechanisms for learning adaptive node representations.
    
    Paper: "Graph Attention Networks"
    """
    
    def __init__(self, heads: int = 8, **kwargs):
        """
        Initialize GAT encoder.
        
        Args:
            heads (int): Number of attention heads
            **kwargs: Base encoder arguments
        """
        self.heads = heads
        super(GATEncoder, self).__init__(**kwargs)
    
    def _build_layers(self):
        """Build GAT layers with multi-head attention."""
        if self.layer_num == 1:
            self.convs.append(GATConv(
                self.input_dim, self.output_dim, 
                heads=1, dropout=self.dropout_rate
            ))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
        else:
            # First layer
            self.convs.append(GATConv(
                self.input_dim, self.hidden_size, 
                heads=self.heads, dropout=self.dropout_rate
            ))
            self.norms.append(NormalizationFactory.get_normalization(
                self.norm_type, self.hidden_size * self.heads
            ))
            
            # Hidden layers
            for _ in range(self.layer_num - 2):
                self.convs.append(GATConv(
                    self.hidden_size * self.heads, self.hidden_size,
                    heads=self.heads, dropout=self.dropout_rate
                ))
                self.norms.append(NormalizationFactory.get_normalization(
                    self.norm_type, self.hidden_size * self.heads
                ))
            
            # Output layer (single head)
            self.convs.append(GATConv(
                self.hidden_size * self.heads, self.output_dim,
                heads=1, dropout=self.dropout_rate
            ))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through GAT layers."""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            
            if i == self.layer_num - 1 and not self.last_activation:
                pass
            else:
                x = self.activation(x)
            
            x = self.dropout(x)
        
        return x


@registry.register_encoder
class GINEncoder(BaseEncoder):
    """
    Graph Isomorphism Network (GIN) encoder.
    
    This encoder implements the Graph Isomorphism Network which is
    theoretically powerful for distinguishing graph structures.
    
    Paper: "How Powerful are Graph Neural Networks?"
    """
    
    def _build_layers(self):
        """Build GIN layers with MLPs."""
        if self.layer_num == 1:
            mlp = Sequential(
                Linear(self.input_dim, self.hidden_size),
                BatchNorm1d(self.hidden_size),
                self.activation,
                Linear(self.hidden_size, self.output_dim)
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
        else:
            # First layer
            mlp = Sequential(
                Linear(self.input_dim, self.hidden_size),
                BatchNorm1d(self.hidden_size),
                self.activation,
                Linear(self.hidden_size, self.hidden_size)
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Hidden layers
            for _ in range(self.layer_num - 2):
                mlp = Sequential(
                    Linear(self.hidden_size, self.hidden_size),
                    BatchNorm1d(self.hidden_size),
                    self.activation,
                    Linear(self.hidden_size, self.hidden_size)
                )
                self.convs.append(GINConv(mlp))
                self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Output layer
            mlp = Sequential(
                Linear(self.hidden_size, self.hidden_size),
                BatchNorm1d(self.hidden_size),
                self.activation,
                Linear(self.hidden_size, self.output_dim)
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through GIN layers."""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            
            if i == self.layer_num - 1 and not self.last_activation:
                pass
            else:
                x = self.activation(x)
            
            x = self.dropout(x)
        
        return x


@registry.register_encoder
class GCNIIEncoder(BaseEncoder):
    """
    Graph Convolutional Network via Initial residual and Identity mapping (GCNII) encoder.
    
    This encoder implements GCNII which addresses over-smoothing in deep GCNs
    through initial residual connections and identity mapping.
    
    Paper: "Simple and Deep Graph Convolutional Networks"
    """
    
    def __init__(self, alpha: float = 0.1, theta: float = 0.5, **kwargs):
        """
        Initialize GCNII encoder.
        
        Args:
            alpha (float): Initial residual parameter
            theta (float): Identity mapping parameter
            **kwargs: Base encoder arguments
        """
        self.alpha = alpha
        self.theta = theta
        super(GCNIIEncoder, self).__init__(**kwargs)
    
    def _build_layers(self):
        """Build GCNII layers."""
        if self.layer_num == 1:
            self.convs.append(GCN2Conv(self.output_dim, alpha=self.alpha, theta=self.theta))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
        else:
            # First layer (standard linear transformation)
            self.input_transform = Linear(self.input_dim, self.hidden_size)
            
            # GCNII layers
            for i in range(self.layer_num):
                layer_alpha = self.alpha
                layer_theta = self.theta / (i + 1) if i > 0 else self.theta
                
                self.convs.append(GCN2Conv(
                    self.hidden_size, alpha=layer_alpha, theta=layer_theta, layer=i+1
                ))
                self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Output transformation
            self.output_transform = Linear(self.hidden_size, self.output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through GCNII layers."""
        if self.layer_num > 1:
            # Initial transformation
            x = self.input_transform(x)
            x0 = x  # Store initial representation
            
            # GCNII layers
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                x = conv(x, x0, edge_index, edge_weight)
                x = norm(x)
                
                if i == self.layer_num - 1 and not self.last_activation:
                    pass
                else:
                    x = self.activation(x)
                
                x = self.dropout(x)
            
            # Output transformation
            x = self.output_transform(x)
        else:
            # Single layer case
            x0 = x
            x = self.convs[0](x, x0, edge_index, edge_weight)
            x = self.norms[0](x)
            
            if self.last_activation:
                x = self.activation(x)
            
            x = self.dropout(x)
        
        return x


@registry.register_encoder
class MLPEncoder(BaseEncoder):
    """
    Multi-Layer Perceptron (MLP) encoder for node features.
    
    This encoder provides a simple baseline using only node features
    without considering graph structure.
    """
    
    def _build_layers(self):
        """Build MLP layers."""
        if self.layer_num == 1:
            self.convs.append(Linear(self.input_dim, self.output_dim))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
        else:
            # First layer
            self.convs.append(Linear(self.input_dim, self.hidden_size))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Hidden layers
            for _ in range(self.layer_num - 2):
                self.convs.append(Linear(self.hidden_size, self.hidden_size))
                self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.hidden_size))
            
            # Output layer
            self.convs.append(Linear(self.hidden_size, self.output_dim))
            self.norms.append(NormalizationFactory.get_normalization(self.norm_type, self.output_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through MLP layers (ignores edge_index)."""
        for i, (linear, norm) in enumerate(zip(self.convs, self.norms)):
            x = linear(x)
            x = norm(x)
            
            if i == self.layer_num - 1 and not self.last_activation:
                pass
            else:
                x = self.activation(x)
            
            x = self.dropout(x)
        
        return x


@registry.register_model
class AdvancedGNN(nn.Module):
    """
    Advanced Graph Neural Network with flexible architecture and pooling options.
    
    This model provides a comprehensive GNN implementation with support for
    multiple encoder types, flexible pooling strategies, and both node-level
    and graph-level predictions.
    
    Features:
    - Multiple encoder architectures via registry system
    - Flexible pooling strategies for graph-level tasks
    - Support for both node and graph classification
    - Efficient subgraph processing
    - Advanced normalization and activation options
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        output_dim: int = 2,
        encoder_type: str = 'GCNEncoder',
        layer_num: int = 2,
        activation: str = 'relu',
        dropout: float = 0.5,
        norm: str = 'batch_norm',
        pooling: str = 'mean',
        last_activation: bool = True,
        use_graph_classifier: bool = True,
        **encoder_kwargs
    ):
        """
        Initialize the advanced GNN model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_size (int): Hidden layer dimension
            output_dim (int): Output dimension (number of classes)
            encoder_type (str): Type of encoder to use
            layer_num (int): Number of encoder layers
            activation (str): Activation function name
            dropout (float): Dropout rate
            norm (str): Normalization type
            pooling (str): Graph pooling strategy
            last_activation (bool): Whether to use activation in last layer
            use_graph_classifier (bool): Whether to use graph-level classifier
            **encoder_kwargs: Additional encoder arguments
        """
        super(AdvancedGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.pooling = pooling
        self.use_graph_classifier = use_graph_classifier
        
        # Create encoder
        self.encoder = registry.get_encoder(
            encoder_type,
            input_dim=input_dim,
            layer_num=layer_num,
            hidden_size=hidden_size,
            output_dim=hidden_size,
            activation=activation,
            dropout=dropout,
            norm=norm,
            last_activation=last_activation,
            **encoder_kwargs
        )
        
        # Node-level classifier
        self.node_classifier = Linear(hidden_size, output_dim)
        
        # Graph-level classifier (for subgraph tasks)
        if use_graph_classifier:
            graph_input_dim = hidden_size * 2  # concatenate node + graph representation
            self.graph_classifier = Linear(graph_input_dim, output_dim)
        
        # Pooling function
        self.pool_fn = self._get_pooling_function(pooling)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"AdvancedGNN initialized: {encoder_type}, "
                   f"{input_dim} -> {hidden_size} -> {output_dim}")
    
    def _get_pooling_function(self, pooling: str) -> Callable:
        """Get pooling function by name."""
        pooling_functions = {
            'mean': global_mean_pool,
            'max': global_max_pool,
            'add': global_add_pool,
            'sum': global_add_pool
        }
        
        if pooling not in pooling_functions:
            logger.warning(f"Unknown pooling '{pooling}', using mean pooling")
            return global_mean_pool
        
        return pooling_functions[pooling]
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.node_classifier.weight)
        nn.init.zeros_(self.node_classifier.bias)
        
        if self.use_graph_classifier:
            nn.init.xavier_uniform_(self.graph_classifier.weight)
            nn.init.zeros_(self.graph_classifier.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        frozen: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for node-level prediction.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_weight (Optional[torch.Tensor]): Edge weights
            frozen (bool): Whether to freeze encoder during forward pass
            
        Returns:
            torch.Tensor: Node-level predictions [num_nodes, output_dim]
        """
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                embeddings = self.encoder(x, edge_index, edge_weight=edge_weight, **kwargs)
        else:
            embeddings = self.encoder(x, edge_index, edge_weight=edge_weight, **kwargs)
        
        # Node-level prediction
        node_predictions = self.node_classifier(embeddings)
        
        return node_predictions
    
    def forward_subgraph(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
        root_node_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for graph-level prediction on subgraphs.
        
        This method is optimized for subgraph classification tasks where
        each subgraph has a designated root node and requires graph-level
        prediction.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            batch (torch.Tensor): Batch assignment for nodes
            root_node_index (torch.Tensor): Indices of root nodes for each graph
            edge_weight (Optional[torch.Tensor]): Edge weights
            
        Returns:
            torch.Tensor: Graph-level predictions [batch_size, output_dim]
        """
        if not self.use_graph_classifier:
            raise RuntimeError("Graph classifier not enabled. Set use_graph_classifier=True")
        
        # Get node embeddings
        embeddings = self.encoder(x, edge_index, edge_weight=edge_weight, **kwargs)
        
        # Get root node embeddings
        root_embeddings = embeddings[root_node_index]
        
        # Get graph-level embeddings through pooling
        graph_embeddings = self.pool_fn(embeddings, batch)
        
        # Concatenate root and graph embeddings
        combined_embeddings = torch.cat([root_embeddings, graph_embeddings], dim=-1)
        
        # Graph-level prediction
        graph_predictions = self.graph_classifier(combined_embeddings)
        
        return graph_predictions
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Get node embeddings without classification.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            edge_weight (Optional[torch.Tensor]): Edge weights
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_size]
        """
        return self.encoder(x, edge_index, edge_weight=edge_weight, **kwargs)
    
    def reset_classifier(self):
        """Reset classifier parameters."""
        self._init_weights()
        logger.debug("Classifier parameters reset")
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.debug("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.debug("Encoder unfrozen")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dict containing model architecture and parameter details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'AdvancedGNN',
            'encoder_type': self.encoder.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_size': self.hidden_size,
            'output_dim': self.output_dim,
            'pooling': self.pooling,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_efficiency': f"{trainable_params/total_params*100:.2f}%",
            'use_graph_classifier': self.use_graph_classifier
        }
        
        return info


# Legacy compatibility - keeping old GNN class for backward compatibility
@registry.register_model
class GNN(AdvancedGNN):
    """Legacy GNN class for backward compatibility."""
    
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=70, 
                 activation="relu", dropout=0.5, norm='id', encoder='GCNEncoder', 
                 last_activation=True, **kwargs):
        """Initialize with legacy parameter names."""
        warnings.warn("GNN class is deprecated. Use AdvancedGNN instead.", 
                     DeprecationWarning, stacklevel=2)
        
        super(GNN, self).__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            encoder_type=encoder + 'Encoder' if not encoder.endswith('Encoder') else encoder,
            layer_num=layer_num,
            activation=activation,
            dropout=dropout,
            norm='batch_norm' if norm == 'bn' else norm,
            last_activation=last_activation,
            **kwargs
        )
        
        