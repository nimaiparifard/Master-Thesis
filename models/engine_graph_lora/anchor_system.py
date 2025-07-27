"""
Anchor Selection and Caching System

This module implements the anchor-based selective refresh mechanism for efficient
LLM computation. It includes anchor selection based on centrality and informativeness,
embedding caching, and prompt routing for budgeted LLM refresh.

Key Features:
- PageRank and degree-based anchor selection
- Text informativeness scoring using entropy
- Efficient embedding caching system
- Prompt router for adaptive LLM refresh
- Memory-mapped storage for large graphs
- Budget-aware refresh scheduling

Author: Graph Neural Network Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import os
import pickle
import mmap
from typing import Dict, List, Tuple, Optional, Union
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class AnchorSelector:
    """
    Anchor selection system for identifying structurally important and text-informative nodes.
    
    This component implements the anchor selection strategy combining structural centrality
    (PageRank, degree) with text informativeness (entropy of neighbor labels) to identify
    the most critical nodes for LLM refresh.
    
    Algorithm:
    1. Compute structural centrality measures (PageRank, degree centrality)
    2. Calculate text informativeness (label entropy in neighborhoods)  
    3. Combine scores with normalized ranking
    4. Select top-K nodes as anchors
    
    Args:
        anchor_ratio (float): Fraction of nodes to select as anchors (0.05-0.25)
        use_pagerank (bool): Whether to include PageRank in selection
        use_degree (bool): Whether to include degree centrality
        use_text_entropy (bool): Whether to include text informativeness
        pagerank_alpha (float): PageRank damping factor
        pagerank_iterations (int): Number of PageRank iterations
        cache_scores (bool): Whether to cache centrality scores
    """
    
    def __init__(
        self,
        anchor_ratio: float = 0.15,
        use_pagerank: bool = True,
        use_degree: bool = True,
        use_text_entropy: bool = True,
        pagerank_alpha: float = 0.85,
        pagerank_iterations: int = 100,
        cache_scores: bool = True
    ):
        self.anchor_ratio = anchor_ratio
        self.use_pagerank = use_pagerank
        self.use_degree = use_degree  
        self.use_text_entropy = use_text_entropy
        self.pagerank_alpha = pagerank_alpha
        self.pagerank_iterations = pagerank_iterations
        self.cache_scores = cache_scores
        
        # Cache for computed scores
        self._score_cache = {}
        
        logger.info(f"AnchorSelector initialized with ratio={anchor_ratio}")
    
    def select_anchors(
        self, 
        edge_index: torch.Tensor,
        node_labels: Optional[torch.Tensor] = None,
        node_texts: Optional[List[str]] = None,
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Select anchor nodes based on structural and textual importance.
        
        Args:
            edge_index (torch.Tensor): Graph edge indices [2, num_edges]
            node_labels (Optional[torch.Tensor]): Node labels for entropy computation
            node_texts (Optional[List[str]]): Node text content for informativeness
            force_recompute (bool): Whether to force recomputation of cached scores
            
        Returns:
            torch.Tensor: Boolean mask indicating anchor nodes [num_nodes]
        """
        num_nodes = edge_index.max().item() + 1
        device = edge_index.device
        
        # Generate cache key
        cache_key = f"anchors_{num_nodes}_{edge_index.shape[1]}_{self.anchor_ratio}"
        
        if not force_recompute and cache_key in self._score_cache:
            logger.info("Using cached anchor selection")
            return self._score_cache[cache_key]
        
        logger.info(f"Computing anchor selection for {num_nodes} nodes")
        
        scores = torch.zeros(num_nodes, device=device)
        
        # 1. Structural centrality scores
        if self.use_pagerank:
            pagerank_scores = self._compute_pagerank(edge_index, num_nodes)
            scores += self._normalize_rank(pagerank_scores)
            logger.debug("Added PageRank scores")
        
        if self.use_degree:
            degree_scores = self._compute_degree_centrality(edge_index, num_nodes)
            scores += self._normalize_rank(degree_scores)
            logger.debug("Added degree centrality scores")
        
        # 2. Text informativeness scores
        if self.use_text_entropy and node_labels is not None:
            entropy_scores = self._compute_label_entropy(edge_index, node_labels, num_nodes)
            scores += self._normalize_rank(entropy_scores)
            logger.debug("Added label entropy scores")
        
        # 3. Select top-K anchors
        k = max(1, int(self.anchor_ratio * num_nodes))
        _, top_indices = torch.topk(scores, k)
        
        anchor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        anchor_mask[top_indices] = True
        
        # Cache results
        if self.cache_scores:
            self._score_cache[cache_key] = anchor_mask
        
        logger.info(f"Selected {k} anchors ({self.anchor_ratio:.1%} of {num_nodes} nodes)")
        
        return anchor_mask
    
    def _compute_pagerank(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute PageRank centrality scores.
        
        Args:
            edge_index (torch.Tensor): Graph edge indices
            num_nodes (int): Number of nodes
            
        Returns:
            torch.Tensor: PageRank scores [num_nodes]
        """
        device = edge_index.device
        
        # Create adjacency matrix
        row, col = edge_index
        edge_weight = torch.ones(edge_index.size(1), device=device)
        
        # Normalize edges
        adj_t = torch.sparse_coo_tensor(
            edge_index, edge_weight, (num_nodes, num_nodes), device=device
        ).coalesce()
        
        # Row normalization for transition matrix
        deg = torch.sparse.sum(adj_t, dim=1).to_dense()
        deg_inv = 1.0 / deg
        deg_inv[deg == 0] = 0
        
        # Initialize PageRank vector
        pr = torch.ones(num_nodes, device=device) / num_nodes
        
        # Power iteration
        for _ in range(self.pagerank_iterations):
            pr_new = (
                self.pagerank_alpha * torch.sparse.mm(adj_t.t(), (pr * deg_inv).unsqueeze(1)).squeeze() + 
                (1 - self.pagerank_alpha) / num_nodes
            )
            
            # Check convergence
            if torch.norm(pr_new - pr) < 1e-6:
                break
            pr = pr_new
        
        return pr
    
    def _compute_degree_centrality(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute degree centrality scores.
        
        Args:
            edge_index (torch.Tensor): Graph edge indices
            num_nodes (int): Number of nodes
            
        Returns:
            torch.Tensor: Degree centrality scores [num_nodes]
        """
        row, col = edge_index
        node_degree = degree(col, num_nodes, dtype=torch.float)
        
        # Normalize by maximum possible degree
        max_degree = num_nodes - 1
        if max_degree > 0:
            node_degree = node_degree / max_degree
        
        return node_degree
    
    def _compute_label_entropy(
        self, 
        edge_index: torch.Tensor, 
        node_labels: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute neighborhood label entropy for text informativeness.
        
        Args:
            edge_index (torch.Tensor): Graph edge indices
            node_labels (torch.Tensor): Node labels [num_nodes]
            num_nodes (int): Number of nodes
            
        Returns:
            torch.Tensor: Label entropy scores [num_nodes]
        """
        device = edge_index.device
        entropies = torch.zeros(num_nodes, device=device)
        
        # Create adjacency list for efficient neighbor lookup
        adj_list = defaultdict(list)
        for i in range(edge_index.size(1)):
            adj_list[edge_index[0, i].item()].append(edge_index[1, i].item())
        
        # Compute entropy for each node's neighborhood
        for node in range(num_nodes):
            neighbors = adj_list[node] + [node]  # Include self
            neighbor_labels = node_labels[neighbors]
            
            # Compute label distribution
            unique_labels, counts = torch.unique(neighbor_labels, return_counts=True)
            probs = counts.float() / counts.sum()
            
            # Compute entropy: H = -Σ p_i * log(p_i)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            entropies[node] = entropy
        
        return entropies
    
    def _normalize_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Normalize scores using rank-based normalization.
        
        Args:
            scores (torch.Tensor): Raw scores
            
        Returns:
            torch.Tensor: Rank-normalized scores in [0, 1]
        """
        # Rank-based normalization as described in the paper
        _, indices = torch.sort(scores, descending=True)
        ranks = torch.zeros_like(scores)
        ranks[indices] = torch.arange(len(scores), dtype=scores.dtype, device=scores.device)
        
        # Normalize to [0, 1]
        normalized = 1.0 - ranks / (len(scores) - 1)
        return normalized


class CacheManager:
    """
    Efficient embedding cache management system.
    
    This component manages the storage and retrieval of node embeddings with support
    for memory-mapped files, LRU eviction, and efficient batch operations.
    
    Features:
    - Memory-mapped storage for large graphs
    - LRU cache with configurable size limits
    - Batch read/write operations
    - Automatic cache persistence and loading
    - Memory usage monitoring and optimization
    
    Args:
        cache_dir (str): Directory for cache storage
        max_memory_mb (int): Maximum memory usage in MB
        embedding_dim (int): Dimension of cached embeddings
        use_mmap (bool): Whether to use memory-mapped files
        compression (bool): Whether to compress cached embeddings
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        max_memory_mb: int = 1024,
        embedding_dim: int = 768,
        use_mmap: bool = True,
        compression: bool = False
    ):
        self.cache_dir = cache_dir
        self.max_memory_mb = max_memory_mb
        self.embedding_dim = embedding_dim
        self.use_mmap = use_mmap
        self.compression = compression
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache with LRU eviction
        self._memory_cache = {}
        self._access_order = []
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Memory-mapped file handle
        self._mmap_file = None
        self._mmap_array = None
        
        logger.info(f"CacheManager initialized: dir={cache_dir}, max_memory={max_memory_mb}MB")
    
    def initialize_cache(self, num_nodes: int, initial_embeddings: Optional[torch.Tensor] = None):
        """
        Initialize cache for a specific number of nodes.
        
        Args:
            num_nodes (int): Total number of nodes to cache
            initial_embeddings (Optional[torch.Tensor]): Initial embeddings to store
        """
        self.num_nodes = num_nodes
        
        if self.use_mmap:
            # Create memory-mapped file
            cache_file = os.path.join(self.cache_dir, f"embeddings_{num_nodes}_{self.embedding_dim}.cache")
            
            # Calculate file size
            file_size = num_nodes * self.embedding_dim * 4  # float32 = 4 bytes
            
            # Create or open memory-mapped file
            if not os.path.exists(cache_file) or os.path.getsize(cache_file) != file_size:
                logger.info(f"Creating cache file: {cache_file} ({file_size / 1024**2:.1f} MB)")
                with open(cache_file, 'wb') as f:
                    f.write(b'\x00' * file_size)
            
            # Open memory-mapped file
            self._cache_file = open(cache_file, 'r+b')
            self._mmap_file = mmap.mmap(self._cache_file.fileno(), 0)
            self._mmap_array = np.frombuffer(self._mmap_file, dtype=np.float32).reshape(
                num_nodes, self.embedding_dim
            )
            
            logger.info(f"Memory-mapped cache initialized: {cache_file}")
        
        # Store initial embeddings
        if initial_embeddings is not None:
            self.update_embeddings(torch.arange(num_nodes), initial_embeddings)
    
    def get_embeddings(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve embeddings for specified nodes.
        
        Args:
            node_indices (torch.Tensor): Node indices to retrieve [batch_size]
            
        Returns:
            torch.Tensor: Retrieved embeddings [batch_size, embedding_dim]
        """
        device = node_indices.device
        batch_size = len(node_indices)
        embeddings = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Convert to numpy for efficient indexing
        indices_np = node_indices.cpu().numpy()
        
        for i, node_idx in enumerate(indices_np):
            # Check memory cache first
            if node_idx in self._memory_cache:
                embeddings[i] = self._memory_cache[node_idx]
                self._update_access_order(node_idx)
                self._cache_hits += 1
            
            # Fall back to memory-mapped file
            elif self._mmap_array is not None:
                embedding = torch.from_numpy(self._mmap_array[node_idx].copy()).to(device)
                embeddings[i] = embedding
                
                # Add to memory cache if space available
                self._maybe_cache_embedding(node_idx, embedding)
                self._cache_misses += 1
            
            else:
                logger.warning(f"No cached embedding found for node {node_idx}")
                self._cache_misses += 1
        
        return embeddings
    
    def update_embeddings(self, node_indices: torch.Tensor, embeddings: torch.Tensor):
        """
        Update cached embeddings for specified nodes.
        
        Args:
            node_indices (torch.Tensor): Node indices to update [batch_size]
            embeddings (torch.Tensor): New embeddings [batch_size, embedding_dim]
        """
        indices_np = node_indices.cpu().numpy()
        embeddings_np = embeddings.detach().cpu().numpy()
        
        for i, node_idx in enumerate(indices_np):
            embedding = embeddings[i]
            
            # Update memory cache
            self._memory_cache[node_idx] = embedding
            self._update_access_order(node_idx)
            
            # Update memory-mapped file
            if self._mmap_array is not None:
                self._mmap_array[node_idx] = embeddings_np[i]
    
    def _maybe_cache_embedding(self, node_idx: int, embedding: torch.Tensor):
        """Add embedding to memory cache if space available."""
        current_memory_mb = len(self._memory_cache) * self.embedding_dim * 4 / (1024**2)
        
        if current_memory_mb < self.max_memory_mb:
            self._memory_cache[node_idx] = embedding
            self._update_access_order(node_idx)
        else:
            # Evict least recently used
            self._evict_lru()
            self._memory_cache[node_idx] = embedding
            self._update_access_order(node_idx)
    
    def _update_access_order(self, node_idx: int):
        """Update LRU access order."""
        if node_idx in self._access_order:
            self._access_order.remove(node_idx)
        self._access_order.append(node_idx)
    
    def _evict_lru(self):
        """Evict least recently used embedding."""
        if self._access_order:
            lru_node = self._access_order.pop(0)
            del self._memory_cache[lru_node]
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self._memory_cache),
            'memory_usage_mb': len(self._memory_cache) * self.embedding_dim * 4 / (1024**2)
        }
    
    def close(self):
        """Clean up resources."""
        if self._mmap_file is not None:
            self._mmap_file.close()
        if hasattr(self, '_cache_file'):
            self._cache_file.close()


class PromptRouter:
    """
    Prompt router for adaptive LLM refresh decisions.
    
    This component implements a learned router that decides which nodes should receive
    LLM refresh based on utility scoring and budget constraints.
    
    Features:
    - Utility prediction based on node and neighborhood features
    - Budget-aware refresh scheduling
    - Stochastic sampling during training
    - Deterministic thresholding during inference
    - Adaptive threshold adjustment based on performance
    
    Args:
        input_dim (int): Input feature dimension for router
        hidden_dims (List[int]): Hidden layer dimensions
        dropout (float): Dropout rate
        tau (float): Temperature for stochastic sampling
        theta_percentile (float): Percentile for deterministic threshold
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        tau: float = 0.5,
        theta_percentile: float = 0.7
    ):
        super(PromptRouter, self).__init__()
        
        self.tau = tau
        self.theta_percentile = theta_percentile
        
        # Build MLP layers
        dims = [input_dim] + hidden_dims + [1]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Running statistics for adaptive thresholding
        self.register_buffer('utility_history', torch.zeros(1000))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
        logger.info(f"PromptRouter initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(
        self,
        node_features: torch.Tensor,
        training: bool = True,
        budget_remaining: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute refresh decisions for nodes.
        
        Args:
            node_features (torch.Tensor): Node features for routing decision [num_nodes, input_dim]
            training (bool): Whether in training mode
            budget_remaining (Optional[float]): Remaining computational budget (0-1)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (refresh_mask, utility_scores)
        """
        # Compute utility scores
        utility_scores = self.mlp(node_features).squeeze(-1)  # [num_nodes]
        
        # Update utility history
        self._update_utility_history(utility_scores)
        
        if training:
            # Stochastic sampling during training
            refresh_probs = torch.sigmoid(utility_scores / self.tau)
            refresh_mask = torch.bernoulli(refresh_probs).bool()
        else:
            # Deterministic thresholding during inference
            threshold = self._compute_adaptive_threshold(budget_remaining)
            refresh_mask = utility_scores > threshold
        
        return refresh_mask, utility_scores
    
    def _update_utility_history(self, utility_scores: torch.Tensor):
        """Update running history of utility scores for adaptive thresholding."""
        with torch.no_grad():
            mean_utility = utility_scores.mean()
            
            ptr = self.history_ptr.item()
            self.utility_history[ptr] = mean_utility
            self.history_ptr[0] = (ptr + 1) % len(self.utility_history)
    
    def _compute_adaptive_threshold(self, budget_remaining: Optional[float] = None) -> float:
        """
        Compute adaptive threshold based on utility history and budget.
        
        Args:
            budget_remaining (Optional[float]): Remaining budget fraction
            
        Returns:
            float: Threshold for refresh decisions
        """
        # Base threshold from utility history percentile
        valid_history = self.utility_history[self.utility_history != 0]
        if len(valid_history) > 0:
            base_threshold = torch.quantile(valid_history, self.theta_percentile).item()
        else:
            base_threshold = 0.0
        
        # Adjust based on remaining budget
        if budget_remaining is not None:
            # Higher threshold when budget is low
            budget_adjustment = (1.0 - budget_remaining) * 0.5
            threshold = base_threshold + budget_adjustment
        else:
            threshold = base_threshold
        
        return threshold
    
    def get_router_stats(self) -> Dict[str, float]:
        """Get router performance statistics."""
        valid_history = self.utility_history[self.utility_history != 0]
        
        if len(valid_history) > 0:
            return {
                'mean_utility': valid_history.mean().item(),
                'std_utility': valid_history.std().item(),
                'current_threshold': self._compute_adaptive_threshold(),
                'history_length': len(valid_history)
            }
        else:
            return {
                'mean_utility': 0.0,
                'std_utility': 0.0,
                'current_threshold': 0.0,
                'history_length': 0
            } 