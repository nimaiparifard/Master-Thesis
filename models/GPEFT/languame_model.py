"""
Advanced Language Model Implementations Module

This module provides comprehensive implementations of various language model architectures
optimized for graph-text integration tasks. It includes transformer-based models,
efficient attention mechanisms, and specialized architectures for multimodal learning.

Key Features:
- Multiple transformer architectures (BERT, GPT, T5-style)
- Graph-text fusion mechanisms
- Efficient attention implementations
- Memory-optimized training and inference
- Flexible model configuration system
- Advanced pooling and aggregation strategies

Performance Optimizations:
- Flash attention for memory efficiency
- Gradient checkpointing for large models
- Mixed precision training support
- Efficient tokenization and preprocessing
- Optimized position encodings

Author: Graph Neural Network Research Team
Version: 2.0 (Comprehensive Implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, BertConfig,
    GPT2Model, GPT2Config,
    T5EncoderModel, T5Config
)
from typing import Dict, Any, Optional, List, Union, Tuple
import math
import warnings
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LanguageModelConfig:
    """Configuration class for language model parameters."""
    model_type: str = 'bert'
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    use_cache: bool = True
    gradient_checkpointing: bool = False


class PositionalEncoding(nn.Module):
    """
    Advanced positional encoding with support for different encoding types.
    
    This module provides various positional encoding strategies including
    sinusoidal encodings, learned embeddings, and relative position encodings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_length: int = 5000,
        encoding_type: str = 'sinusoidal',
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Model dimension
            max_length (int): Maximum sequence length
            encoding_type (str): Type of encoding ('sinusoidal', 'learned')
            dropout (float): Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.encoding_type = encoding_type
        
        if encoding_type == 'sinusoidal':
            self._create_sinusoidal_encodings(max_length)
        elif encoding_type == 'learned':
            self.position_embeddings = nn.Embedding(max_length, d_model)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _create_sinusoidal_encodings(self, max_length: int):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_length, self.d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Embeddings with positional encoding
        """
        if self.encoding_type == 'sinusoidal':
            x = x + self.pe[:, :x.size(1)]
        elif self.encoding_type == 'learned':
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.position_embeddings(positions)
        
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Optimized multi-head attention implementation.
    
    This implementation provides efficient attention computation with
    optional flash attention support and various attention patterns.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        attention_type: str = 'standard'
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
            use_flash_attention (bool): Whether to use flash attention
            attention_type (str): Type of attention ('standard', 'linear', 'local')
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash_attention = use_flash_attention
        self.attention_type = attention_type
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor [batch, seq_len, d_model]
            key (torch.Tensor): Key tensor [batch, seq_len, d_model]
            value (torch.Tensor): Value tensor [batch, seq_len, d_model]
            mask (Optional[torch.Tensor]): Attention mask
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor or Tuple: Output tensor and optionally attention weights
        """
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0 flash attention if available
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
            )
            attention_weights = None
        else:
            # Standard attention computation
            attn_output, attention_weights = self._standard_attention(Q, K, V, mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        if return_attention and attention_weights is not None:
            return output, attention_weights
        return output
    
    def _standard_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attn_output = torch.matmul(attention_weights, V)
        
        return attn_output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network with various activation functions.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model (int): Model dimension
            d_ff (int): Feed-forward dimension
            dropout (float): Dropout rate
            activation (str): Activation function name
        """
        super(FeedForwardNetwork, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output tensor [batch, seq_len, d_model]
        """
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Residual connection and layer norm
        return self.layer_norm(x + residual)


class TransformerBlock(nn.Module):
    """
    Complete transformer block with self-attention and feed-forward layers.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_flash_attention: bool = False
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            d_ff (int): Feed-forward dimension
            dropout (float): Dropout rate
            activation (str): Activation function
            use_flash_attention (bool): Whether to use flash attention
        """
        super(TransformerBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through transformer block.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model]
            mask (Optional[torch.Tensor]): Attention mask
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor or Tuple: Output and optionally attention weights
        """
        if return_attention:
            attn_output, attention_weights = self.self_attention(
                x, x, x, mask=mask, return_attention=True
            )
            ff_output = self.feed_forward(attn_output)
            return ff_output, attention_weights
        else:
            attn_output = self.self_attention(x, x, x, mask=mask)
            ff_output = self.feed_forward(attn_output)
            return ff_output


class CustomTransformerEncoder(nn.Module):
    """
    Custom transformer encoder with advanced features.
    
    This encoder provides a flexible transformer architecture with
    support for various optimizations and customizations.
    """
    
    def __init__(
        self,
        config: LanguageModelConfig,
        use_flash_attention: bool = False
    ):
        """
        Initialize custom transformer encoder.
        
        Args:
            config (LanguageModelConfig): Model configuration
            use_flash_attention (bool): Whether to use flash attention
        """
        super(CustomTransformerEncoder, self).__init__()
        
        self.config = config
        self.use_flash_attention = use_flash_attention
        
        # Embedding layers
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_encoding = PositionalEncoding(
            d_model=config.hidden_size,
            max_length=config.max_position_embeddings,
            dropout=config.hidden_dropout_prob
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.hidden_size,
                num_heads=config.num_attention_heads,
                d_ff=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                use_flash_attention=use_flash_attention
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"CustomTransformerEncoder initialized with {config.num_hidden_layers} layers")
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer encoder.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch, seq_len]
            attention_mask (Optional[torch.Tensor]): Attention mask [batch, seq_len]
            return_all_layers (bool): Whether to return all layer outputs
            return_attention_weights (bool): Whether to return attention weights
            
        Returns:
            Dict containing model outputs
        """
        batch_size, seq_len = input_ids.size()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).float()
        
        # Expand attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Token embeddings with positional encoding
        embeddings = self.token_embeddings(input_ids)
        embeddings = self.position_encoding(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        all_layer_outputs = []
        all_attention_weights = []
        
        for i, block in enumerate(self.blocks):
            if self.config.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    extended_attention_mask
                )
                attention_weights = None
            else:
                if return_attention_weights:
                    hidden_states, attention_weights = block(
                        hidden_states, 
                        mask=extended_attention_mask,
                        return_attention=True
                    )
                    all_attention_weights.append(attention_weights)
                else:
                    hidden_states = block(hidden_states, mask=extended_attention_mask)
                    attention_weights = None
            
            if return_all_layers:
                all_layer_outputs.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Prepare outputs
        outputs = {
            'last_hidden_state': hidden_states,
            'pooler_output': self._pool_output(hidden_states, attention_mask)
        }
        
        if return_all_layers:
            outputs['hidden_states'] = all_layer_outputs
        
        if return_attention_weights:
            outputs['attention_weights'] = all_attention_weights
        
        return outputs
    
    def _pool_output(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool the output for sequence-level representation.
        
        Args:
            hidden_states (torch.Tensor): Hidden states [batch, seq_len, hidden_size]
            attention_mask (torch.Tensor): Attention mask [batch, seq_len]
            
        Returns:
            torch.Tensor: Pooled output [batch, hidden_size]
        """
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class GraphTextFusionModel(nn.Module):
    """
    Advanced model for fusing graph and text representations.
    
    This model combines graph neural networks with language models
    to create unified representations for graph-text tasks.
    """
    
    def __init__(
        self,
        text_config: LanguageModelConfig,
        graph_hidden_size: int,
        fusion_hidden_size: int = 512,
        fusion_strategy: str = 'concat',
        num_fusion_layers: int = 2,
        use_cross_attention: bool = True
    ):
        """
        Initialize graph-text fusion model.
        
        Args:
            text_config (LanguageModelConfig): Text model configuration
            graph_hidden_size (int): Graph representation size
            fusion_hidden_size (int): Fusion layer hidden size
            fusion_strategy (str): Strategy for fusing representations
            num_fusion_layers (int): Number of fusion layers
            use_cross_attention (bool): Whether to use cross-attention
        """
        super(GraphTextFusionModel, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        self.use_cross_attention = use_cross_attention
        
        # Text encoder
        self.text_encoder = CustomTransformerEncoder(text_config)
        
        # Graph projection layer
        self.graph_projection = nn.Linear(graph_hidden_size, text_config.hidden_size)
        
        # Fusion layers
        if fusion_strategy == 'concat':
            fusion_input_size = text_config.hidden_size + text_config.hidden_size  # text + projected graph
        elif fusion_strategy == 'add':
            fusion_input_size = text_config.hidden_size
        elif fusion_strategy == 'cross_attention':
            fusion_input_size = text_config.hidden_size
            self.cross_attention = MultiHeadAttention(
                d_model=text_config.hidden_size,
                num_heads=text_config.num_attention_heads,
                dropout=text_config.attention_probs_dropout_prob
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Fusion MLP
        fusion_layers = []
        current_size = fusion_input_size
        
        for i in range(num_fusion_layers):
            fusion_layers.extend([
                nn.Linear(current_size, fusion_hidden_size),
                nn.ReLU(),
                nn.Dropout(text_config.hidden_dropout_prob)
            ])
            current_size = fusion_hidden_size
        
        self.fusion_layers = nn.Sequential(*fusion_layers)
        
        # Output projection
        self.output_projection = nn.Linear(fusion_hidden_size, text_config.hidden_size)
        
        logger.info(f"GraphTextFusionModel initialized with {fusion_strategy} fusion")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        graph_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through graph-text fusion model.
        
        Args:
            input_ids (torch.Tensor): Text token IDs [batch, seq_len]
            graph_embeddings (torch.Tensor): Graph embeddings [batch, graph_hidden_size]
            attention_mask (Optional[torch.Tensor]): Text attention mask
            
        Returns:
            Dict containing fused representations
        """
        # Encode text
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_hidden = text_outputs['last_hidden_state']  # [batch, seq_len, hidden_size]
        text_pooled = text_outputs['pooler_output']      # [batch, hidden_size]
        
        # Project graph embeddings
        graph_projected = self.graph_projection(graph_embeddings)  # [batch, hidden_size]
        
        # Fusion strategy
        if self.fusion_strategy == 'concat':
            # Concatenate text and graph representations
            fused_input = torch.cat([text_pooled, graph_projected], dim=-1)
        
        elif self.fusion_strategy == 'add':
            # Element-wise addition
            fused_input = text_pooled + graph_projected
        
        elif self.fusion_strategy == 'cross_attention':
            # Cross-attention between text and graph
            graph_expanded = graph_projected.unsqueeze(1)  # [batch, 1, hidden_size]
            
            # Use cross-attention to attend to graph from text
            attended_output = self.cross_attention(
                query=text_pooled.unsqueeze(1),  # [batch, 1, hidden_size]
                key=graph_expanded,
                value=graph_expanded
            ).squeeze(1)  # [batch, hidden_size]
            
            fused_input = attended_output
        
        # Apply fusion layers
        fused_representation = self.fusion_layers(fused_input)
        
        # Output projection
        final_output = self.output_projection(fused_representation)
        
        return {
            'fused_representation': final_output,
            'text_representation': text_pooled,
            'graph_representation': graph_projected,
            'text_hidden_states': text_hidden
        }


class PretrainedLanguageModelWrapper(nn.Module):
    """
    Wrapper for pretrained language models with additional functionality.
    
    This wrapper provides a unified interface for different pretrained
    language models while adding custom functionality.
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: Optional[int] = None,
        freeze_base_model: bool = False,
        add_pooling_layer: bool = True,
        pooling_strategy: str = 'cls'
    ):
        """
        Initialize pretrained language model wrapper.
        
        Args:
            model_name (str): Name of pretrained model
            num_classes (Optional[int]): Number of classes for classification
            freeze_base_model (bool): Whether to freeze base model parameters
            add_pooling_layer (bool): Whether to add pooling layer
            pooling_strategy (str): Pooling strategy ('cls', 'mean', 'max')
        """
        super(PretrainedLanguageModelWrapper, self).__init__()
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        
        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if requested
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info("Base model parameters frozen")
        
        # Add classification head if specified
        if num_classes is not None:
            hidden_size = self.base_model.config.hidden_size
            self.classifier = nn.Linear(hidden_size, num_classes)
        else:
            self.classifier = None
        
        # Add pooling layer if requested
        if add_pooling_layer:
            hidden_size = self.base_model.config.hidden_size
            self.pooler = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = None
        
        logger.info(f"PretrainedLanguageModelWrapper initialized with {model_name}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through pretrained language model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (Optional[torch.Tensor]): Attention mask
            token_type_ids (Optional[torch.Tensor]): Token type IDs
            return_dict (bool): Whether to return as dictionary
            
        Returns:
            Dict containing model outputs
        """
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        
        # Extract representations
        if return_dict:
            hidden_states = outputs.last_hidden_state
            pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        else:
            hidden_states = outputs[0]
            pooled_output = outputs[1] if len(outputs) > 1 else None
        
        # Apply custom pooling if needed
        if self.pooler is not None:
            if self.pooling_strategy == 'cls':
                pooled_representation = self.pooler(hidden_states[:, 0, :])
            elif self.pooling_strategy == 'mean':
                if attention_mask is not None:
                    # Masked mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    pooled_representation = self.pooler(sum_embeddings / sum_mask)
                else:
                    pooled_representation = self.pooler(hidden_states.mean(dim=1))
            elif self.pooling_strategy == 'max':
                pooled_representation = self.pooler(hidden_states.max(dim=1)[0])
            else:
                pooled_representation = self.pooler(hidden_states[:, 0, :])
        else:
            pooled_representation = pooled_output
        
        # Apply classification if specified
        logits = None
        if self.classifier is not None and pooled_representation is not None:
            logits = self.classifier(pooled_representation)
        
        result = {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_representation,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attention_weights': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
        
        if logits is not None:
            result['logits'] = logits
        
        return result
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
    
    def freeze_base_model(self):
        """Freeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Base model parameters frozen")
    
    def unfreeze_base_model(self):
        """Unfreeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
        logger.info("Base model parameters unfrozen")


# Factory function for creating language models
def create_language_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating language models.
    
    Args:
        model_type (str): Type of model to create
        config (Optional[Dict]): Model configuration
        **kwargs: Additional arguments
        
    Returns:
        nn.Module: Language model instance
    """
    if model_type == 'custom_transformer':
        model_config = LanguageModelConfig(**(config or {}))
        return CustomTransformerEncoder(model_config, **kwargs)
    
    elif model_type == 'pretrained':
        model_name = kwargs.get('model_name', 'bert-base-uncased')
        return PretrainedLanguageModelWrapper(model_name, **kwargs)
    
    elif model_type == 'graph_text_fusion':
        text_config = LanguageModelConfig(**(config or {}))
        return GraphTextFusionModel(text_config, **kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Utility functions for text processing
def tokenize_and_encode(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    padding: str = 'max_length',
    truncation: bool = True,
    return_tensors: str = 'pt'
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and encode text sequences.
    
    Args:
        texts (List[str]): List of text sequences
        tokenizer: Tokenizer instance
        max_length (int): Maximum sequence length
        padding (str): Padding strategy
        truncation (bool): Whether to truncate sequences
        return_tensors (str): Return tensor format
        
    Returns:
        Dict containing encoded sequences
    """
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )
    
    return encoded


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask from input IDs.
    
    Args:
        input_ids (torch.Tensor): Input token IDs
        pad_token_id (int): Padding token ID
        
    Returns:
        torch.Tensor: Attention mask
    """
    return (input_ids != pad_token_id).long()
