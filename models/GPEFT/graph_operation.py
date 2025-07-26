from torch_geometric.loader import DataLoader

"""
Graph Structure Operations Module

This module provides efficient graph manipulation utilities for PyTorch Geometric graphs,
including subgraph sampling, node isolation handling, and ego graph generation.
The module focuses on performance optimization and memory efficiency for large-scale
graph neural network applications.

Key Features:
- Self-loop addition for isolated nodes
- Random walk-based subgraph collection
- Efficient ego graph sampling
- Node index remapping utilities
- Advanced subsampling strategies for scalable training

Author: Graph Neural Network Research Team
Version: 2.0 (Optimized)
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected,
    k_hop_subgraph,
    to_edge_index
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from typing import List, Tuple, Union, Optional, Dict, Any
import warnings
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)


def add_remaining_selfloop_for_isolated_nodes(
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None
) -> torch.Tensor:
    """
    Add self-loops only to isolated nodes in the graph.

    This function identifies nodes that have no connections (isolated nodes) and adds
    self-loops to them. This is crucial for graph neural networks as isolated nodes
    might not receive proper message passing updates without self-connections.

    Performance Optimizations:
    - Uses vectorized operations instead of loops
    - Leverages boolean masking for efficient node filtering
    - Minimizes tensor concatenations

    Args:
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges] representing
                                 the graph connectivity in COO format.
        num_nodes (Optional[int]): Total number of nodes in the graph. If None,
                                 inferred from edge_index.

    Returns:
        torch.Tensor: Modified edge_index with self-loops added to isolated nodes.
                     Shape: [2, num_edges + num_isolated_nodes]

    Example:
        >>> edge_index = torch.tensor([[0, 1], [1, 0]])  # Only nodes 0-1 connected
        >>> num_nodes = 4  # Nodes 2, 3 are isolated
        >>> result = add_remaining_selfloop_for_isolated_nodes(edge_index, num_nodes)
        >>> # Result includes self-loops for nodes 2 and 3

    Time Complexity: O(E + N) where E is number of edges, N is number of nodes
    Space Complexity: O(N) for the boolean mask
    """
    # Ensure we have the correct number of nodes
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index)
    else:
        num_nodes = max(maybe_num_nodes(edge_index), num_nodes)

    # Early return if no nodes
    if num_nodes == 0:
        return edge_index

    # Get all connected nodes efficiently using unique operation
    connected_nodes = torch.cat([edge_index[0], edge_index[1]]).unique()

    # Create boolean mask for isolated nodes (vectorized operation)
    isolated_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    isolated_mask[connected_nodes] = False

    # Get isolated node indices
    isolated_nodes = torch.nonzero(isolated_mask, as_tuple=True)[0]

    # Early return if no isolated nodes
    if isolated_nodes.numel() == 0:
        return edge_index

    # Create self-loops for isolated nodes efficiently
    self_loops = isolated_nodes.unsqueeze(0).repeat(2, 1)

    # Concatenate original edges with self-loops
    enhanced_edge_index = torch.cat([edge_index, self_loops], dim=1)

    return enhanced_edge_index


def collect_subgraphs(
        selected_nodes: torch.Tensor,
        graph: Data,
        walk_steps: int = 20,
        restart_ratio: float = 0.5
) -> List[Data]:
    """
    Collect subgraphs using random walk with restart sampling strategy.

    This function performs random walks starting from selected nodes to generate
    subgraphs. The random walk with restart mechanism helps capture both local
    neighborhood information and maintains connection to the starting node,
    which is essential for node-centric graph learning tasks.

    Algorithm Details:
    1. Start random walks from each selected node
    2. At each step, with probability `restart_ratio`, return to the start node
    3. Otherwise, move to a random neighbor
    4. Collect all visited nodes to form subgraphs
    5. Create proper PyG Data objects with re-indexed nodes

    Performance Optimizations:
    - Uses sparse tensor operations for memory efficiency
    - Vectorized random walk sampling across multiple starting nodes
    - Pre-allocates tensors to minimize memory reallocation
    - Efficient tensor operations for path tracking

    Args:
        selected_nodes (torch.Tensor): Node indices to start random walks from.
                                     Shape: [num_start_nodes]
        graph (Data): PyTorch Geometric Data object containing the full graph
        walk_steps (int): Number of steps in each random walk. Default: 20
        restart_ratio (float): Probability of restarting to the original node
                              at each step. Range: [0, 1]. Default: 0.5

    Returns:
        List[Data]: List of subgraph Data objects, one for each starting node.
                   Each contains:
                   - x: Node features for subgraph nodes
                   - edge_index: Edges within the subgraph
                   - y: Label of the center node
                   - center: Index of center node in subgraph
                   - original_idx: Original node indices
                   - root_n_index: Center node index (for PyG compatibility)

    Raises:
        ValueError: If restart_ratio is not in [0, 1]
        RuntimeError: If selected_nodes contains invalid indices

    Example:
        >>> selected_nodes = torch.tensor([0, 5, 10])
        >>> subgraphs = collect_subgraphs(selected_nodes, graph, walk_steps=15)
        >>> len(subgraphs)  # Returns 3 subgraphs
        3

    Time Complexity: O(W * S * D) where W is walk_steps, S is len(selected_nodes),
                     D is average node degree
    Space Complexity: O(W * S) for storing walk history
    """
    # Input validation
    if not 0 <= restart_ratio <= 1:
        raise ValueError(f"restart_ratio must be in [0, 1], got {restart_ratio}")

    if selected_nodes.max() >= graph.x.shape[0]:
        raise RuntimeError("selected_nodes contains invalid node indices")

    # Use graph directly without deep copy for better performance
    edge_index = graph.edge_index
    num_nodes = graph.x.shape[0]
    num_start_nodes = selected_nodes.shape[0]

    # Convert to sparse tensor for efficient sampling
    if isinstance(edge_index, SparseTensor):
        adj_sparse = edge_index
    else:
        # Create sparse adjacency matrix
        edge_values = torch.arange(edge_index.size(1), device=edge_index.device)
        adj_sparse = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_values,
            sparse_sizes=(num_nodes, num_nodes)
        ).t()

    # Initialize walk tracking tensors
    current_nodes = selected_nodes.clone()

    # Pre-allocate tensors for better performance
    walk_history = torch.zeros(
        (walk_steps + 1, num_start_nodes),
        dtype=torch.long,
        device=selected_nodes.device
    )
    restart_flags = torch.zeros(
        (walk_steps + 1, num_start_nodes),
        dtype=torch.bool,
        device=selected_nodes.device
    )

    # Set initial state
    walk_history[0] = selected_nodes
    restart_flags[0] = True

    # Perform vectorized random walks
    for step in range(walk_steps):
        # Generate random restart decisions
        restart_decisions = torch.rand(num_start_nodes, device=selected_nodes.device) < restart_ratio

        # Sample neighbors for all current nodes simultaneously
        try:
            next_nodes = adj_sparse.sample(1, current_nodes).squeeze()
        except Exception:
            # Fallback for nodes with no neighbors
            next_nodes = current_nodes.clone()

        # Apply restart decisions
        next_nodes[restart_decisions] = selected_nodes[restart_decisions]

        # Update tracking tensors
        walk_history[step + 1] = next_nodes
        restart_flags[step + 1] = restart_decisions
        current_nodes = next_nodes

    # Generate subgraphs for each starting node
    subgraph_list = []

    for i in range(num_start_nodes):
        node_path = walk_history[:, i]
        restart_mask = restart_flags[:, i]

        # Get unique nodes visited in this walk
        unique_nodes = node_path.unique()

        # Create edges from the walk path (excluding restart steps)
        valid_steps = ~restart_mask[1:]  # Exclude restart transitions
        if valid_steps.any():
            path_sources = node_path[:-1][valid_steps]
            path_targets = node_path[1:][valid_steps]

            if len(path_sources) > 0:
                subgraph_edges = torch.stack([path_sources, path_targets])
                # Make undirected for better connectivity
                subgraph_edges = to_undirected(subgraph_edges)
            else:
                # Create empty edge tensor with correct shape
                subgraph_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        else:
            subgraph_edges = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        # Create subgraph data object
        subgraph = _create_subgraph_data(
            subgraph_edges, unique_nodes, graph, selected_nodes[i].item()
        )
        subgraph_list.append(subgraph)

    return subgraph_list


def _create_subgraph_data(
        edge_index: torch.Tensor,
        node_indices: torch.Tensor,
        full_graph: Data,
        center_node_id: int
) -> Data:
    """
    Create a PyTorch Geometric Data object for a subgraph with proper node re-indexing.

    This helper function handles the complex task of creating a valid PyG Data object
    from a subgraph by re-mapping node indices to create a contiguous index space
    while preserving the original graph structure and node features.

    Performance Optimizations:
    - Uses vectorized tensor operations for index mapping
    - Leverages advanced indexing for efficient feature extraction
    - Minimizes Python loops in favor of tensor operations

    Args:
        edge_index (torch.Tensor): Edge indices in the original node space
        node_indices (torch.Tensor): Original node indices to include in subgraph
        full_graph (Data): Original graph data object
        center_node_id (int): ID of the center node in original indexing

    Returns:
        Data: PyTorch Geometric Data object with re-indexed nodes and edges

    Time Complexity: O(E + N) where E is edges, N is nodes in subgraph
    Space Complexity: O(N) for index mapping
    """
    # Create mapping from original indices to new contiguous indices
    node_mapping = torch.zeros(full_graph.x.shape[0], dtype=torch.long, device=node_indices.device)
    node_mapping[node_indices] = torch.arange(len(node_indices), device=node_indices.device)

    # Re-index edges if any exist
    if edge_index.numel() > 0:
        reindexed_edges = node_mapping[edge_index]
    else:
        reindexed_edges = edge_index

    # Extract subgraph features
    subgraph_features = full_graph.x[node_indices]

    # Find center node in new indexing
    center_idx_new = node_mapping[center_node_id].item()

    # Create Data object with all necessary attributes
    subgraph_data = Data(
        x=subgraph_features,
        edge_index=reindexed_edges,
        y=full_graph.y[center_node_id],
        center=center_idx_new,
        original_idx=node_indices,
        root_n_index=center_idx_new  # PyG compatibility
    )

    return subgraph_data


def ego_graphs_sampler(
        node_indices: torch.Tensor,
        graph_data: Data,
        num_hops: int = 2,
        use_sparse: bool = False
) -> List[Data]:
    """
    Generate ego graphs (k-hop subgraphs) centered on specified nodes.

    An ego graph for a node includes all nodes within k hops of the center node
    and all edges between these nodes. This is a fundamental operation in graph
    neural networks for creating node-centric subgraphs that capture local
    neighborhood structure.

    Performance Optimizations:
    - Leverages PyG's optimized k_hop_subgraph function
    - Uses batch processing where possible
    - Efficient tensor indexing for feature extraction
    - Optional sparse tensor support for memory efficiency

    Algorithm:
    1. For each target node, extract k-hop neighborhood using PyG utilities
    2. Create subgraph with proper node relabeling
    3. Preserve node features and maintain center node information
    4. Handle edge cases (isolated nodes, boundary nodes)

    Args:
        node_indices (torch.Tensor): Indices of nodes to create ego graphs for.
                                   Shape: [num_target_nodes]
        graph_data (Data): PyTorch Geometric Data object containing the full graph
        num_hops (int): Number of hops to include in each ego graph. Default: 2
        use_sparse (bool): Whether to convert sparse tensors to edge_index format.
                          Default: False

    Returns:
        List[Data]: List of ego graph Data objects, one for each input node.
                   Each contains:
                   - x: Node features for ego graph nodes
                   - edge_index: Edges within the ego graph
                   - y: Label of the center node
                   - root_n_index: Center node index in ego graph
                   - original_idx: Original node indices from full graph

    Raises:
        ValueError: If num_hops is negative
        RuntimeError: If node_indices contains invalid indices

    Example:
        >>> node_indices = torch.tensor([0, 10, 20])
        >>> ego_graphs = ego_graphs_sampler(node_indices, graph_data, num_hops=2)
        >>> # Returns 3 ego graphs, each containing 2-hop neighborhoods

    Time Complexity: O(N * D^k) where N is len(node_indices), D is average degree,
                     k is num_hops
    Space Complexity: O(D^k) per ego graph
    """
    # Input validation
    if num_hops < 0:
        raise ValueError(f"num_hops must be non-negative, got {num_hops}")

    if node_indices.max() >= graph_data.x.shape[0]:
        raise RuntimeError("node_indices contains invalid node indices")

    # Handle sparse tensor conversion if needed
    if use_sparse and hasattr(graph_data, 'edge_index'):
        try:
            edge_index, _ = to_edge_index(graph_data.edge_index)
        except Exception:
            edge_index = graph_data.edge_index
            warnings.warn("Failed to convert sparse tensor, using original edge_index")
    else:
        edge_index = graph_data.edge_index

    ego_graph_list = []

    # Generate ego graph for each target node
    for target_node in node_indices:
        try:
            # Extract k-hop subgraph using PyG's optimized function
            subset_nodes, subgraph_edges, center_mapping, _ = k_hop_subgraph(
                node_idx=[target_node.item()],
                num_hops=num_hops,
                edge_index=edge_index,
                relabel_nodes=True
            )

            # Extract node features for the subgraph
            subgraph_features = graph_data.x[subset_nodes]

            # Create ego graph Data object
            ego_graph = Data(
                x=subgraph_features,
                edge_index=subgraph_edges,
                y=graph_data.y[target_node],
                root_n_index=center_mapping[0],  # Center node index in subgraph
                original_idx=subset_nodes
            )

            ego_graph_list.append(ego_graph)

        except Exception as e:
            # Handle edge cases (e.g., isolated nodes)
            warnings.warn(f"Failed to create ego graph for node {target_node}: {e}")

            # Create minimal ego graph with just the target node
            ego_graph = Data(
                x=graph_data.x[target_node].unsqueeze(0),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                y=graph_data.y[target_node],
                root_n_index=torch.tensor([0]),
                original_idx=target_node.unsqueeze(0)
            )
            ego_graph_list.append(ego_graph)

    return ego_graph_list


def analyze_subgraph_statistics(subgraphs: List[Data]) -> Dict[str, Union[int, float]]:
    """
    Analyze statistics of generated subgraphs for debugging and optimization.

    This function provides comprehensive statistics about subgraph collections,
    helping to understand the quality and characteristics of the sampling process.

    Args:
        subgraphs (List[Data]): List of subgraph Data objects

    Returns:
        Dict: Statistics including size distributions, connectivity metrics

    Example:
        >>> subgraphs = collect_subgraphs(selected_nodes, graph)
        >>> stats = analyze_subgraph_statistics(subgraphs)
        >>> print(f"Average nodes per subgraph: {stats['avg_nodes']:.2f}")
    """
    if not subgraphs:
        return {"error": "No subgraphs provided"}

    num_nodes = [g.x.shape[0] for g in subgraphs]
    num_edges = [g.edge_index.shape[1] for g in subgraphs]

    # Calculate connectivity metrics
    avg_degree = [2 * edges / max(nodes, 1) for nodes, edges in zip(num_nodes, num_edges)]
    
    stats = {
        "num_subgraphs": len(subgraphs),
        "avg_nodes": sum(num_nodes) / len(num_nodes),
        "avg_edges": sum(num_edges) / len(num_edges),
        "min_nodes": min(num_nodes),
        "max_nodes": max(num_nodes),
        "min_edges": min(num_edges),
        "max_edges": max(num_edges),
        "avg_degree": sum(avg_degree) / len(avg_degree),
        "node_std": torch.tensor(num_nodes, dtype=torch.float).std().item(),
        "edge_std": torch.tensor(num_edges, dtype=torch.float).std().item(),
    }

    return stats


def subsampling(
    data: Data, 
    config: Any, 
    sampler: str = 'rw',
    return_stats: bool = False
) -> Union[Tuple[DataLoader, DataLoader, DataLoader], 
           Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]]:
    """
    Advanced subgraph sampling for scalable GNN training.
    
    This function implements multiple subgraph sampling strategies to enable
    training on large graphs that don't fit in GPU memory. It provides
    flexible sampling options and comprehensive performance optimizations.
    
    Sampling Strategies:
    - 'rw': Random Walk with Restart - captures local neighborhoods with global context
    - 'khop': K-hop neighborhood sampling - includes all nodes within k hops
    - 'fastgcn': FastGCN-style sampling (if implemented)
    - 'graphsaint': GraphSAINT-style sampling (if implemented)
    
    Performance Optimizations:
    - Persistent workers for reduced DataLoader overhead
    - Optimized batch size and worker configuration
    - Memory-efficient subgraph generation
    - Parallel subgraph processing where possible
    
    Args:
        data (Data): PyTorch Geometric Data object containing the full graph
        config (Any): Configuration object containing sampling parameters:
            - walk_steps (int): Number of random walk steps (for 'rw')
            - restart (float): Restart probability (for 'rw')
            - k (int): Number of hops (for 'khop')
            - batch_size (int): Batch size for DataLoaders
        sampler (str): Sampling strategy ('rw' or 'khop'). Default: 'rw'
        return_stats (bool): Whether to return sampling statistics. Default: False
    
    Returns:
        Union[Tuple, Tuple with stats]: DataLoaders for train/val/test splits,
                                       optionally with sampling statistics
    
    Raises:
        ValueError: If sampler type is not supported
        RuntimeError: If sampling fails due to graph structure issues
    
    Example:
        >>> train_loader, val_loader, test_loader = subsampling(data, config, 'rw')
        >>> # With statistics
        >>> loaders_and_stats = subsampling(data, config, 'rw', return_stats=True)
        >>> train_loader, val_loader, test_loader, stats = loaders_and_stats
    
    Time Complexity: O(N * S) where N=nodes, S=sampling_complexity
    Space Complexity: O(B * S_avg) where B=batch_size, S_avg=avg_subgraph_size
    """
    start_time = time.time()
    
    # Input validation
    if sampler not in ['rw', 'khop']:
        raise ValueError(f"Unsupported sampler type: {sampler}. Use 'rw' or 'khop'")
    
    # Extract node indices for different splits
    try:
        train_idx = data.train_mask.nonzero().squeeze()
        val_idx = data.val_mask.nonzero().squeeze()
        test_idx = data.test_mask.nonzero().squeeze()
    except AttributeError:
        raise RuntimeError("Data object must have train_mask, val_mask, and test_mask attributes")
    
    logger.info(f"Starting {sampler} subgraph sampling")
    logger.info(f"Train nodes: {len(train_idx)}, Val nodes: {len(val_idx)}, Test nodes: {len(test_idx)}")
    
    # Initialize sampling statistics
    sampling_stats = {
        'sampler_type': sampler,
        'total_nodes': data.x.shape[0],
        'total_edges': data.edge_index.shape[1],
        'sampling_time': 0.0,
        'train_subgraphs': len(train_idx),
        'val_subgraphs': len(val_idx),
        'test_subgraphs': len(test_idx)
    }
    
    # Perform subgraph sampling based on strategy
    if sampler == 'rw':
        # Random Walk with Restart sampling
        walk_steps = getattr(config, 'walk_steps', 20)
        restart_ratio = getattr(config, 'restart', 0.5)
        
        logger.info(f"Random walk parameters: steps={walk_steps}, restart_ratio={restart_ratio}")
        
        train_graphs = collect_subgraphs(
            train_idx, data, walk_steps=walk_steps, restart_ratio=restart_ratio
        )
        val_graphs = collect_subgraphs(
            val_idx, data, walk_steps=walk_steps, restart_ratio=restart_ratio
        )
        test_graphs = collect_subgraphs(
            test_idx, data, walk_steps=walk_steps, restart_ratio=restart_ratio
        )
        
        # Update sampling statistics
        sampling_stats.update({
            'walk_steps': walk_steps,
            'restart_ratio': restart_ratio
        })
        
    elif sampler == 'khop':
        # K-hop neighborhood sampling
        k_hops = getattr(config, 'k', 2)
        
        logger.info(f"K-hop parameters: k={k_hops}")
        
        train_graphs = ego_graphs_sampler(train_idx, data, num_hops=k_hops)
        val_graphs = ego_graphs_sampler(val_idx, data, num_hops=k_hops)
        test_graphs = ego_graphs_sampler(test_idx, data, num_hops=k_hops)
        
        # Update sampling statistics
        sampling_stats.update({
            'k_hops': k_hops
        })
    
    # Calculate sampling time
    sampling_time = time.time() - start_time
    sampling_stats['sampling_time'] = sampling_time
    
    logger.info(f"Subgraph sampling completed in {sampling_time:.2f} seconds")
    
    # Analyze subgraph statistics if requested
    if return_stats:
        train_stats = analyze_subgraph_statistics(train_graphs)
        val_stats = analyze_subgraph_statistics(val_graphs)
        test_stats = analyze_subgraph_statistics(test_graphs)
        
        sampling_stats.update({
            'train_stats': train_stats,
            'val_stats': val_stats,
            'test_stats': test_stats
        })
    
    # Configure DataLoader parameters for optimal performance
    batch_size = getattr(config, 'batch_size', 32)
    num_workers = min(getattr(config, 'num_workers', 6), 8)  # Cap workers for memory efficiency
    
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'persistent_workers': True if num_workers > 0 else False,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': True  # For consistent batch sizes
    }
    
    # Create DataLoaders with optimized settings
    train_loader = DataLoader(train_graphs, shuffle=True, **dataloader_kwargs)
    
    # Use larger batch size for evaluation (no gradients needed)
    eval_kwargs = dataloader_kwargs.copy()
    eval_kwargs['batch_size'] = min(batch_size * 2, 128)  # Larger eval batches
    eval_kwargs['shuffle'] = False
    eval_kwargs['drop_last'] = False
    
    val_loader = DataLoader(val_graphs, **eval_kwargs)
    test_loader = DataLoader(test_graphs, **eval_kwargs)
    
    logger.info(f"DataLoaders created with batch_size={batch_size}, "
               f"eval_batch_size={eval_kwargs['batch_size']}, num_workers={num_workers}")
    
    if return_stats:
        return train_loader, val_loader, test_loader, sampling_stats
    else:
        return train_loader, val_loader, test_loader


def adaptive_subsampling(
    data: Data,
    config: Any,
    node_importance: Optional[torch.Tensor] = None,
    dynamic_sizing: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Adaptive subgraph sampling with importance-based node selection.
    
    This advanced sampling strategy adapts the subgraph generation process
    based on node importance scores and graph characteristics. It can
    dynamically adjust subgraph sizes and sampling probabilities.
    
    Args:
        data (Data): Graph data
        config (Any): Configuration object
        node_importance (Optional[torch.Tensor]): Importance scores for nodes
        dynamic_sizing (bool): Whether to use dynamic subgraph sizing
    
    Returns:
        Tuple of DataLoaders for train/val/test
    """
    # Implementation for adaptive sampling
    # This is a placeholder for advanced sampling strategies
    logger.info("Using adaptive subsampling strategy")
    
    # Fall back to standard subsampling for now
    return subsampling(data, config, sampler='rw')


def validate_graph_structure(data: Data) -> Dict[str, Union[bool, int, float]]:
    """
    Validate graph structure and return diagnostic information.
    
    Args:
        data (Data): Graph data to validate
    
    Returns:
        Dict containing validation results and statistics
    """
    validation_results = {
        'is_valid': True,
        'num_nodes': data.x.shape[0],
        'num_edges': data.edge_index.shape[1],
        'has_self_loops': False,
        'has_isolated_nodes': False,
        'is_undirected': False,
        'max_degree': 0,
        'avg_degree': 0.0
    }
    
    # Check for self-loops
    self_loop_mask = data.edge_index[0] == data.edge_index[1]
    validation_results['has_self_loops'] = self_loop_mask.any().item()
    
    # Check for isolated nodes
    connected_nodes = torch.cat([data.edge_index[0], data.edge_index[1]]).unique()
    validation_results['has_isolated_nodes'] = len(connected_nodes) < data.x.shape[0]
    
    # Calculate degree statistics
    degrees = torch.bincount(data.edge_index[0], minlength=data.x.shape[0])
    validation_results['max_degree'] = degrees.max().item()
    validation_results['avg_degree'] = degrees.float().mean().item()
    
    # Check if undirected (approximate)
    edge_set_1 = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    edge_set_2 = set(zip(data.edge_index[1].tolist(), data.edge_index[0].tolist()))
    validation_results['is_undirected'] = edge_set_1 == edge_set_2
    
    return validation_results