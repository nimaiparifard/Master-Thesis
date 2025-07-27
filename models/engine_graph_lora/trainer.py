"""
ENGINE-GraphLoRA Trainer: Comprehensive Training Framework

This module implements the complete training workflow for ENGINE-GraphLoRA,
including offline preprocessing, anchor selection, multi-objective optimization,
and evaluation protocols.

Author: Graph Neural Network Research Team
Version: 1.0
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_scheduler
import wandb
from tqdm import tqdm

from .config import EngineGraphLoRAConfig
from .model import EngineGraphLoRAModel
from .anchor_system import AnchorSelector, CacheManager
# Removed external imports - will implement graph utility functions locally

logger = logging.getLogger(__name__)


# ===== LOCAL UTILITY FUNCTIONS =====

def compute_pagerank(edge_index: torch.Tensor, num_nodes: int, alpha: float = 0.85, max_iter: int = 100) -> torch.Tensor:
    """
    Compute PageRank scores for graph nodes.
    
    Args:
        edge_index: Graph edges [2, num_edges]
        num_nodes: Number of nodes in the graph
        alpha: Damping factor (default: 0.85)
        max_iter: Maximum iterations (default: 100)
        
    Returns:
        PageRank scores [num_nodes]
    """
    try:
        from torch_geometric.utils import degree
        
        # Create adjacency matrix
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        
        # Simple power iteration approximation
        pr = torch.ones(num_nodes) / num_nodes
        
        for _ in range(max_iter):
            pr_new = (1 - alpha) / num_nodes + alpha * torch.zeros(num_nodes)
            for i in range(len(row)):
                pr_new[col[i]] += alpha * pr[row[i]] * deg_inv[row[i]]
            
            if torch.allclose(pr, pr_new, atol=1e-6):
                break
            pr = pr_new
            
        return pr
    except Exception as e:
        logger.warning(f"PageRank computation failed: {e}, using uniform scores")
        return torch.ones(num_nodes) / num_nodes


def compute_degree_centrality(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute degree centrality for graph nodes.
    
    Args:
        edge_index: Graph edges [2, num_edges]
        num_nodes: Number of nodes in the graph
        
    Returns:
        Degree centrality scores [num_nodes]
    """
    try:
        from torch_geometric.utils import degree
        
        # Compute degree for each node
        deg = degree(edge_index[0], num_nodes, dtype=torch.float)
        deg += degree(edge_index[1], num_nodes, dtype=torch.float)
        
        # Normalize by maximum possible degree
        max_degree = max(deg.max().item(), 1.0)
        centrality = deg / max_degree
        
        return centrality
    except Exception as e:
        logger.warning(f"Degree centrality computation failed: {e}, using uniform scores")
        return torch.ones(num_nodes) / num_nodes


def compute_clustering_coefficient(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute clustering coefficient for graph nodes.
    
    Args:
        edge_index: Graph edges [2, num_edges]
        num_nodes: Number of nodes in the graph
        
    Returns:
        Clustering coefficient scores [num_nodes]
    """
    try:
        from torch_geometric.utils import to_dense_adj
        
        # Convert to adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        
        # Compute clustering coefficient
        clustering = torch.zeros(num_nodes)
        
        for i in range(num_nodes):
            neighbors = adj[i].nonzero().squeeze()
            if len(neighbors) < 2:
                clustering[i] = 0.0
                continue
                
            # Count triangles
            subadj = adj[neighbors][:, neighbors]
            triangles = subadj.sum().item() / 2
            
            # Possible edges between neighbors
            k = len(neighbors)
            possible = k * (k - 1) / 2
            
            clustering[i] = triangles / possible if possible > 0 else 0.0
            
        return clustering
    except Exception as e:
        logger.warning(f"Clustering coefficient computation failed: {e}, using zeros")
        return torch.zeros(num_nodes)


class EngineGraphLoRATrainer:
    """
    Comprehensive trainer for ENGINE-GraphLoRA model with full workflow implementation.
    
    Features:
    - Offline preprocessing with text embedding cache and centrality profiling
    - Anchor-based selective LLM refresh with budgeting
    - Multi-objective optimization with SMMD, contrastive, and structural losses
    - Early-exit inference with entropy-based stopping
    - Comprehensive evaluation and logging
    """
    
    def __init__(
        self,
        config: EngineGraphLoRAConfig,
        model: EngineGraphLoRAModel,
        train_data: Any,
        val_data: Any,
        test_data: Optional[Any] = None,
        device: Optional[str] = None
    ):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.device = device or config.device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        
        # Anchor and cache systems
        self.anchor_selector = None
        self.cache_manager = None
        self._anchor_nodes = None
        
        # Offline preprocessing state
        self._preprocessed = False
        self._text_cache_built = False
        self._centrality_computed = False
        
        logger.info("Initialized ENGINE-GraphLoRA trainer")
        
    def _setup_optimizer(self):
        """Setup optimizer for trainable parameters only"""
        # Get only trainable parameters (LoRA adapters, G-Ladder, etc.)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.training.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
            
        logger.info(f"Setup {self.config.training.optimizer} optimizer with {len(trainable_params)} trainable parameter groups")
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        num_training_steps = len(self.train_data) * self.config.training.num_epochs
        
        self.scheduler = get_scheduler(
            name=self.config.training.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def _setup_logging(self):
        """Setup experiment tracking and logging"""
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.run_name or f"{self.config.experiment_name}_{int(time.time())}",
                config=self.config.to_dict()
            )
            
        # Create checkpoint directory
        self.checkpoint_dir = Path(f"checkpoints/{self.config.experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def offline_preprocessing(self, data: Any) -> Dict[str, Any]:
        """
        Perform offline preprocessing: text embedding cache + centrality profiling
        
        Args:
            data: Graph data with node texts and structure
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        logger.info("Starting offline preprocessing...")
        
        # ===== STEP 1: Text Embedding Cache =====
        if not self._text_cache_built:
            logger.info("Building text embedding cache...")
            cache_stats = self._build_text_cache(data)
            self._text_cache_built = True
            logger.info(f"Text cache built: {cache_stats}")
        
        # ===== STEP 2: Centrality Profiling =====
        if not self._centrality_computed:
            logger.info("Computing graph centralities...")
            centrality_stats = self._compute_centralities(data)
            self._centrality_computed = True
            logger.info(f"Centralities computed: {centrality_stats}")
        
        # ===== STEP 3: Anchor Node Selection =====
        logger.info("Selecting anchor nodes...")
        anchor_stats = self._select_anchor_nodes(data)
        logger.info(f"Anchor selection: {anchor_stats}")
        
        self._preprocessed = True
        
        preprocessing_results = {
            'cache_stats': cache_stats if self._text_cache_built else {},
            'centrality_stats': centrality_stats if self._centrality_computed else {},
            'anchor_stats': anchor_stats,
            'anchor_nodes': self._anchor_nodes
        }
        
        logger.info("Offline preprocessing completed")
        return preprocessing_results
        
    def _build_text_cache(self, data: Any) -> Dict[str, Any]:
        """Build text embedding cache using frozen LLM"""
        cache_manager = CacheManager(
            cache_dir=self.config.llm.cache_dir,
            cache_format=self.config.llm.cache_format,
            hidden_size=self.config.llm.hidden_size
        )
        
        # Process texts in batches
        batch_size = 32
        num_nodes = len(data.x) if hasattr(data, 'x') else data.num_nodes
        cached_count = 0
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, num_nodes, batch_size), desc="Caching embeddings"):
                batch_end = min(i + batch_size, num_nodes)
                batch_indices = torch.arange(i, batch_end)
                
                # Get text data for batch
                if hasattr(data, 'text'):
                    batch_texts = [data.text[idx] for idx in batch_indices]
                else:
                    # Fallback to node features or dummy text
                    batch_texts = [f"Node {idx}" for idx in batch_indices]
                
                # Tokenize
                inputs = self.model.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.data.max_text_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings from LLM
                llm_outputs = self.model.llm(**inputs, output_hidden_states=True)
                
                # Mean pooling over sequence
                embeddings = llm_outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                # Cache embeddings
                cache_manager.update_embeddings(batch_indices, embeddings.cpu())
                cached_count += len(batch_indices)
        
        self.model.cache_manager = cache_manager
        
        return {
            'total_nodes': num_nodes,
            'cached_nodes': cached_count,
            'cache_size_mb': cache_manager.get_cache_size() / (1024 * 1024)
        }
        
    def _compute_centralities(self, data: Any) -> Dict[str, Any]:
        """Compute graph centrality measures"""
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Compute PageRank
        pagerank_scores = compute_pagerank(edge_index, num_nodes)
        
        # Compute degree centrality
        degree_scores = compute_degree_centrality(edge_index, num_nodes)
        
        # Compute clustering coefficient
        clustering_scores = compute_clustering_coefficient(edge_index, num_nodes)
        
        # Store centralities in data object
        data.pagerank = pagerank_scores
        data.degree = degree_scores
        data.clustering = clustering_scores
        
        return {
            'pagerank_mean': float(pagerank_scores.mean()),
            'pagerank_std': float(pagerank_scores.std()),
            'degree_mean': float(degree_scores.mean()),
            'degree_std': float(degree_scores.std()),
            'clustering_mean': float(clustering_scores.mean()),
            'clustering_std': float(clustering_scores.std())
        }
        
    def _select_anchor_nodes(self, data: Any) -> Dict[str, Any]:
        """Select anchor nodes based on centrality and text informativeness"""
        if not hasattr(self, 'anchor_selector') or self.anchor_selector is None:
            self.anchor_selector = AnchorSelector(
                anchor_ratio=self.config.anchor.anchor_ratio,
                selection_method=self.config.anchor.selection_method,
                pagerank_weight=self.config.anchor.pagerank_weight,
                degree_weight=self.config.anchor.degree_weight,
                entropy_weight=self.config.anchor.entropy_weight
            )
        
        # Prepare features for anchor selection
        features = {
            'pagerank': data.pagerank,
            'degree': data.degree,
            'clustering': data.clustering
        }
        
        # Add text entropy if available
        if hasattr(data, 'y'):
            # Compute label entropy in neighborhood
            features['label_entropy'] = self._compute_label_entropy(data.edge_index, data.y, data.num_nodes)
        
        # Select anchors
        anchor_indices = self.anchor_selector.select_anchors(features, data.num_nodes)
        self._anchor_nodes = anchor_indices
        
        # Create anchor mask
        anchor_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        anchor_mask[anchor_indices] = True
        data.anchor_mask = anchor_mask
        
        return {
            'total_anchors': len(anchor_indices),
            'anchor_ratio': len(anchor_indices) / data.num_nodes,
            'anchor_pagerank_mean': float(data.pagerank[anchor_indices].mean()),
            'anchor_degree_mean': float(data.degree[anchor_indices].mean())
        }
        
    def _compute_label_entropy(self, edge_index: torch.Tensor, labels: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute label entropy in node neighborhoods"""
        entropies = torch.zeros(num_nodes)
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for src, dst in edge_index.t():
            adj_list[src.item()].append(dst.item())
            
        for node in range(num_nodes):
            neighbors = adj_list[node]
            if len(neighbors) == 0:
                entropies[node] = 0.0
                continue
                
            # Get neighbor labels
            neighbor_labels = labels[neighbors]
            
            # Compute label distribution
            unique_labels, counts = torch.unique(neighbor_labels, return_counts=True)
            probs = counts.float() / len(neighbors)
            
            # Compute entropy
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            entropies[node] = entropy
            
        return entropies
        
    def train(self) -> Dict[str, List[float]]:
        """
        Execute the complete training workflow
        
        Returns:
            Dictionary containing training metrics history
        """
        if not self._preprocessed:
            logger.info("Running offline preprocessing...")
            self.offline_preprocessing(self.train_data)
        
        logger.info("Starting training...")
        
        # Training metrics
        train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.model.setup_for_training()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Validation epoch
            if (epoch + 1) % self.config.training.validation_frequency == 0:
                val_metrics = self._validate_epoch()
                
                # Update history
                train_history['val_loss'].append(val_metrics['loss'])
                train_history['val_acc'].append(val_metrics['accuracy'])
                
                # Early stopping check
                if val_metrics['accuracy'] > self.best_val_score:
                    self.best_val_score = val_metrics['accuracy']
                    self.patience_counter = 0
                    self._save_checkpoint('best_model.pt')
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                val_metrics = {}
            
            # Update history
            train_history['train_loss'].append(train_metrics['total_loss'])
            train_history['train_acc'].append(train_metrics['accuracy'])
            train_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        logger.info("Training completed")
        return train_history
        
    def _train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch"""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'task_loss': 0.0,
            'smmd_loss': 0.0,
            'contrastive_loss': 0.0,
            'structure_loss': 0.0,
            'accuracy': 0.0,
            'num_batches': 0,
            'refresh_rate': 0.0
        }
        
        total_correct = 0
        total_samples = 0
        total_refreshes = 0
        total_nodes = 0
        
        # Training loop with anchor batching - Implementing 4-step algorithm
        for batch_idx, batch in enumerate(tqdm(self._get_anchor_batches(), desc=f"Epoch {self.current_epoch + 1}")):
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # ============= 4-STEP ALGORITHM IMPLEMENTATION =============
            
            # ➊ SPARSE LLM REFRESH (budgeted) - only on anchors
            anchor_mask = batch.get('anchor_mask')
            h_llm = self._selective_llm_refresh(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                anchor_mask=anchor_mask,
                budget_ratio=self.config.anchor.anchor_budget_ratio
            )
            
            # Update cache with refreshed embeddings
            if anchor_mask is not None and h_llm is not None:
                refresh_indices = anchor_mask.nonzero().squeeze(-1)
                self.model.cache_manager.update_embeddings(refresh_indices, h_llm.detach().cpu())
            
            # ➋ CROSS-LAYER FUSION - G-Ladder modules
            h_fused_layers = []
            h_current = h_llm
            
            # Apply G-Ladder fusion
            h_current = self.model.g_ladder(h_current, batch['edge_index'])
            h_fused_layers.append(h_current)
            
            # ➌ MESSAGE PASSING - GNN with LoRA-SAGE
            # Standard GNN processing
            z_graph = self.model.gnn(h_current, batch['edge_index'])
            
            # Apply LoRA adaptation (low-rank ΔW)
            z_lora = self.model.lora_sage(z_graph, batch['edge_index'])
            
            # ➍ MULTI-OBJECTIVE LOSS - 𝓛_task + λ·𝓛_SMMD + μ·𝓛_contrast
            # Get final predictions
            logits = self.model.prediction_head(z_lora)
            
            # Compute individual loss components
            task_loss = F.cross_entropy(logits, batch['labels'])
            
            # SMMD loss between text and graph representations
            smmd_loss = self.model.smmd_loss(
                source_features=h_current,  # text representations
                target_features=z_lora,     # graph representations
                source_edge_index=batch['edge_index'],
                target_edge_index=batch['edge_index']
            )
            
            # Contrastive loss for text-graph alignment
            contrastive_loss = self.model.contrastive_loss(
                text_representations=h_current,
                graph_representations=z_lora
            )
            
            # Combine losses with weights
            λ = self.config.loss.smmd_loss_weight
            μ = self.config.loss.contrastive_loss_weight
            
            total_loss = task_loss + λ * smmd_loss + μ * contrastive_loss
            
            # ============= BACKPROP (only adapters) =============
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Store outputs for metrics computation
            outputs = {
                'logits': logits,
                'h_llm': h_llm,
                'h_fused': h_current,
                'z_graph': z_graph,
                'z_lora': z_lora
            }
            
            losses = {
                'total_loss': total_loss,
                'task_loss': task_loss,
                'smmd_loss': smmd_loss,
                'contrastive_loss': contrastive_loss
            }
            
            # Update metrics
            batch_size = len(batch['labels'])
            predictions = outputs['logits'].argmax(dim=1)
            correct = (predictions == batch['labels']).sum().item()
            
            epoch_metrics['total_loss'] += losses['total_loss'].item()
            epoch_metrics['task_loss'] += losses['task_loss'].item()
            epoch_metrics['smmd_loss'] += losses['smmd_loss'].item()
            epoch_metrics['contrastive_loss'] += losses['contrastive_loss'].item()
            epoch_metrics['num_batches'] += 1
            
            total_correct += correct
            total_samples += batch_size
            
            # Track refresh rate
            if 'anchor_mask' in batch and batch['anchor_mask'] is not None:
                total_refreshes += batch['anchor_mask'].sum().item()
                total_nodes += len(batch['anchor_mask'])
            
            self.global_step += 1
        
        # Compute average metrics
        num_batches = max(epoch_metrics['num_batches'], 1)
        for key in ['total_loss', 'task_loss', 'smmd_loss', 'contrastive_loss']:
            epoch_metrics[key] /= num_batches
        
        epoch_metrics['accuracy'] = total_correct / max(total_samples, 1)
        epoch_metrics['refresh_rate'] = total_refreshes / max(total_nodes, 1)
        
        return epoch_metrics
        
    def _validate_epoch(self) -> Dict[str, float]:
        """Execute validation epoch"""
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'num_batches': 0
        }
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self._get_validation_batches(), desc="Validation"):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    edge_index=batch['edge_index'],
                    batch_nodes=batch.get('batch_nodes')
                )
                
                # Compute loss
                task_loss = F.cross_entropy(outputs['logits'], batch['labels'])
                
                # Update metrics
                predictions = outputs['logits'].argmax(dim=1)
                correct = (predictions == batch['labels']).sum().item()
                batch_size = len(batch['labels'])
                
                val_metrics['loss'] += task_loss.item()
                val_metrics['num_batches'] += 1
                
                total_correct += correct
                total_samples += batch_size
        
        # Compute averages
        num_batches = max(val_metrics['num_batches'], 1)
        val_metrics['loss'] /= num_batches
        val_metrics['accuracy'] = total_correct / max(total_samples, 1)
        
        return val_metrics
        
    def _get_anchor_batches(self):
        """Generate batches focused on anchor nodes"""
        # Implementation depends on your data structure
        # This is a placeholder that should be adapted to your specific data format
        anchor_indices = self._anchor_nodes
        batch_size = self.config.training.batch_size
        
        # Create batches of anchor nodes
        for i in range(0, len(anchor_indices), batch_size):
            batch_anchors = anchor_indices[i:i + batch_size]
            
            # Prepare batch data
            batch = self._prepare_batch(batch_anchors)
            yield batch
    
    def _get_validation_batches(self):
        """Generate validation batches"""
        # Implementation placeholder
        batch_size = self.config.training.batch_size
        num_val_nodes = len(self.val_data.y) if hasattr(self.val_data, 'y') else self.val_data.num_nodes
        
        for i in range(0, num_val_nodes, batch_size):
            batch_nodes = torch.arange(i, min(i + batch_size, num_val_nodes))
            batch = self._prepare_batch(batch_nodes, is_validation=True)
            yield batch
    
    def _prepare_batch(self, node_indices: torch.Tensor, is_validation: bool = False):
        """Prepare batch data for given node indices"""
        # This is a placeholder that should be implemented based on your data structure
        data = self.val_data if is_validation else self.train_data
        
        # Extract text data
        if hasattr(data, 'text'):
            texts = [data.text[idx] for idx in node_indices]
        else:
            texts = [f"Node {idx}" for idx in node_indices]
        
        # Tokenize
        inputs = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_text_length,
            return_tensors='pt'
        )
        
        batch = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'edge_index': data.edge_index,
            'labels': data.y[node_indices],
            'batch_nodes': node_indices
        }
        
        if hasattr(data, 'anchor_mask') and not is_validation:
            batch['anchor_mask'] = data.anchor_mask[node_indices]
        
        return batch
    
    def _selective_llm_refresh(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor_mask: Optional[torch.Tensor],
        budget_ratio: float
    ) -> torch.Tensor:
        """
        Implement selective LLM refresh with budget constraints.
        
        This implements step ➊ of the algorithm: sparse LLM refresh (budgeted)
        """
        device = input_ids.device
        num_nodes = input_ids.size(0)
        
        # If no anchor mask provided, treat all nodes as potential anchors
        if anchor_mask is None:
            anchor_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        
        # Determine budget for this batch
        anchor_indices = anchor_mask.nonzero().squeeze(-1)
        budget_size = max(1, int(len(anchor_indices) * budget_ratio))
        
        # Select which anchors to refresh based on importance/uncertainty
        if len(anchor_indices) > budget_size:
            # For now, randomly select anchors (in practice, use centrality/uncertainty)
            selected_indices = anchor_indices[torch.randperm(len(anchor_indices))[:budget_size]]
        else:
            selected_indices = anchor_indices
        
        # Initialize with cached embeddings
        h_llm = torch.zeros(num_nodes, self.config.llm.hidden_size, device=device)
        
        # Get cached embeddings for non-refreshed nodes
        non_refresh_indices = torch.ones(num_nodes, dtype=torch.bool, device=device)
        non_refresh_indices[selected_indices] = False
        
        if non_refresh_indices.any():
            cached_embeddings = self.model.cache_manager.get_embeddings(
                non_refresh_indices.nonzero().squeeze(-1)
            )
            if cached_embeddings is not None:
                h_llm[non_refresh_indices] = cached_embeddings.to(device)
        
        # Refresh selected anchors through LLM
        if len(selected_indices) > 0:
            refresh_input_ids = input_ids[selected_indices]
            refresh_attention_mask = attention_mask[selected_indices]
            
            with torch.cuda.amp.autocast(enabled=self.config.training.use_fp16):
                llm_outputs = self.model.llm(
                    input_ids=refresh_input_ids,
                    attention_mask=refresh_attention_mask,
                    output_hidden_states=True
                )
            
            # Mean pooling over sequence length
            hidden_states = llm_outputs.last_hidden_state
            pooled_embeddings = (hidden_states * refresh_attention_mask.unsqueeze(-1)).sum(dim=1) / \
                               refresh_attention_mask.sum(dim=1, keepdim=True)
            
            h_llm[selected_indices] = pooled_embeddings
        
        return h_llm

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log training metrics"""
        # Console logging
        log_str = f"Epoch {epoch + 1}/{self.config.training.num_epochs}"
        log_str += f" | Train Loss: {train_metrics['total_loss']:.4f}"
        log_str += f" | Train Acc: {train_metrics['accuracy']:.4f}"
        
        if val_metrics:
            log_str += f" | Val Loss: {val_metrics['loss']:.4f}"
            log_str += f" | Val Acc: {val_metrics['accuracy']:.4f}"
        
        log_str += f" | Refresh Rate: {train_metrics['refresh_rate']:.3f}"
        log_str += f" | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        
        logger.info(log_str)
        
        # WandB logging
        if self.config.wandb_project:
            log_dict = {
                'epoch': epoch,
                'train/total_loss': train_metrics['total_loss'],
                'train/task_loss': train_metrics['task_loss'],
                'train/smmd_loss': train_metrics['smmd_loss'],
                'train/contrastive_loss': train_metrics['contrastive_loss'],
                'train/structure_loss': train_metrics['structure_loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/refresh_rate': train_metrics['refresh_rate'],
                'train/learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            if val_metrics:
                log_dict.update({
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy']
                })
            
            wandb.log(log_dict, step=self.global_step)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config.to_dict(),
            'anchor_nodes': self._anchor_nodes
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_score = checkpoint['best_val_score']
        self._anchor_nodes = checkpoint.get('anchor_nodes')
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def evaluate(self, data: Any = None) -> Dict[str, float]:
        """Evaluate model on test data"""
        if data is None:
            data = self.test_data
        
        if data is None:
            raise ValueError("No test data provided")
        
        self.model.setup_for_inference()
        
        eval_metrics = {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'avg_exit_layer': 0.0,
            'inference_time': 0.0
        }
        
        total_correct = 0
        total_samples = 0
        total_exit_layers = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self._get_test_batches(data), desc="Evaluation"):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    edge_index=batch['edge_index'],
                    batch_nodes=batch.get('batch_nodes')
                )
                
                # Compute metrics
                predictions = outputs['logits'].argmax(dim=1)
                correct = (predictions == batch['labels']).sum().item()
                
                total_correct += correct
                total_samples += len(batch['labels'])
                
                # Track early exit
                if 'exit_layer' in outputs:
                    total_exit_layers += outputs['exit_layer'].sum().item()
        
        eval_metrics['accuracy'] = total_correct / max(total_samples, 1)
        eval_metrics['avg_exit_layer'] = total_exit_layers / max(total_samples, 1)
        eval_metrics['inference_time'] = time.time() - start_time
        
        logger.info(f"Evaluation results: {eval_metrics}")
        
        return eval_metrics
    
    def _get_test_batches(self, data):
        """Generate test batches"""
        batch_size = self.config.inference.inference_batch_size
        num_test_nodes = len(data.y) if hasattr(data, 'y') else data.num_nodes
        
        for i in range(0, num_test_nodes, batch_size):
            batch_nodes = torch.arange(i, min(i + batch_size, num_test_nodes))
            batch = self._prepare_batch(batch_nodes, is_validation=True)
            yield batch 