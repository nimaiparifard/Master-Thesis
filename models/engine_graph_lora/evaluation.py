"""
ENGINE-GraphLoRA Evaluation Module

This module provides comprehensive evaluation utilities specifically designed for
ENGINE-GraphLoRA models with multi-objective optimization, early-exit capabilities,
and anchor-based selective processing.

Key Features:
- Multi-objective evaluation with task, SMMD, and contrastive losses
- Anchor-based evaluation with selective LLM refresh
- Early-exit inference with confidence-based stopping
- Comprehensive metrics including efficiency measurements
- Support for both standard and budget-aware evaluation modes
- Memory-efficient batch processing with gradient-free inference

Performance Optimizations:
- Vectorized operations for batch processing
- Memory-efficient tensor operations with proper cleanup
- Cache-aware evaluation to minimize LLM computations
- Dynamic evaluation scheduling based on anchor budgets

Author: Graph Neural Network Research Team
Version: 1.0 (Optimized for ENGINE-GraphLoRA)
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time
from collections import defaultdict
import numpy as np
import logging
from pathlib import Path

from .model import EngineGraphLoRAModel
from .config import EngineGraphLoRAConfig
from .arguments import EngineGraphLoRAArguments

logger = logging.getLogger(__name__)


class EngineGraphLoRAEvaluator:
    """
    Comprehensive evaluator for ENGINE-GraphLoRA models.
    
    This evaluator implements sophisticated evaluation strategies including:
    - Multi-objective loss evaluation (task + SMMD + contrastive)
    - Anchor-based selective evaluation with budget constraints
    - Early-exit inference with entropy-based confidence thresholding
    - Efficiency metrics including computational cost and memory usage
    - Comprehensive performance analysis across different evaluation modes
    """
    
    def __init__(
        self,
        model: EngineGraphLoRAModel,
        config: EngineGraphLoRAConfig,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = device or config.device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Evaluation state
        self.evaluation_mode = 'standard'  # 'standard', 'early_exit', 'budget_aware'
        self.cache_hit_rate = 0.0
        self.total_evaluations = 0
        
        # Metrics tracking
        self.detailed_metrics = defaultdict(list)
        self.efficiency_metrics = defaultdict(list)
        
        logger.info(f"Initialized ENGINE-GraphLoRA evaluator on {self.device}")
    
    def comprehensive_evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        evaluation_modes: List[str] = ['standard', 'early_exit', 'budget_aware'],
        return_detailed_metrics: bool = True,
        save_predictions: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive evaluation across multiple modes.
        
        This function evaluates the ENGINE-GraphLoRA model using different strategies
        to provide a complete performance analysis including accuracy, efficiency,
        and resource utilization metrics.
        
        Args:
            data_loader: DataLoader containing evaluation data
            evaluation_modes: List of evaluation modes to run
            return_detailed_metrics: Whether to compute detailed performance metrics
            save_predictions: Whether to save model predictions
            
        Returns:
            Dictionary containing results for each evaluation mode
        """
        logger.info(f"Starting comprehensive evaluation with modes: {evaluation_modes}")
        
        results = {}
        
        for mode in evaluation_modes:
            logger.info(f"Evaluating in {mode} mode...")
            self.evaluation_mode = mode
            
            if mode == 'standard':
                mode_results = self.evaluate_standard(
                    data_loader, return_detailed_metrics, save_predictions
                )
            elif mode == 'early_exit':
                mode_results = self.evaluate_early_exit(
                    data_loader, return_detailed_metrics, save_predictions
                )
            elif mode == 'budget_aware':
                mode_results = self.evaluate_budget_aware(
                    data_loader, return_detailed_metrics, save_predictions
                )
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")
            
            results[mode] = mode_results
            
            # Log key metrics
            logger.info(f"{mode} mode results:")
            logger.info(f"  Accuracy: {mode_results.get('accuracy', 0.0):.4f}")
            logger.info(f"  Total Loss: {mode_results.get('total_loss', 0.0):.4f}")
            if 'avg_computational_cost' in mode_results:
                logger.info(f"  Avg Computational Cost: {mode_results['avg_computational_cost']:.2f}")
        
        return results
    
    def evaluate_standard(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_detailed_metrics: bool = True,
        save_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Standard evaluation without early exit or budget constraints.
        
        This function evaluates the complete ENGINE-GraphLoRA model using the full
        architecture for all samples. It provides baseline performance metrics
        and represents the maximum accuracy potential of the model.
        
        Args:
            data_loader: DataLoader containing evaluation data
            return_detailed_metrics: Whether to compute detailed metrics
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        # Initialize metrics tracking
        total_correct = 0
        total_samples = 0
        total_losses = defaultdict(float)
        predictions_list = []
        labels_list = []
        
        # Detailed metrics
        if return_detailed_metrics:
            confidence_scores = []
            processing_times = []
            memory_usage = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Standard Evaluation")):
                batch_start_time = time.time()
                
                # Move batch to device
                batch = self._prepare_batch(batch)
                batch_size = self._get_batch_size(batch)
                total_samples += batch_size
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    edge_index=batch['edge_index'],
                    node_features=batch.get('node_features'),
                    batch_nodes=batch.get('batch_nodes'),
                    anchor_mask=batch.get('anchor_mask')
                )
                
                # Compute losses
                losses = self.model.compute_loss(
                    outputs=outputs,
                    labels=batch['labels'],
                    edge_index=batch['edge_index'],
                    source_data=batch.get('source_data'),
                    target_data=batch.get('target_data')
                )
                
                # Update loss tracking
                for loss_name, loss_value in losses.items():
                    total_losses[loss_name] += loss_value.item()
                
                # Compute accuracy
                predictions = outputs['logits'].argmax(dim=1)
                correct = (predictions == batch['labels']).sum().item()
                total_correct += correct
                
                # Store predictions and labels
                if save_predictions:
                    predictions_list.extend(predictions.cpu().numpy())
                    labels_list.extend(batch['labels'].cpu().numpy())
                
                # Detailed metrics
                if return_detailed_metrics:
                    # Confidence scores
                    probs = F.softmax(outputs['logits'], dim=1)
                    max_probs = probs.max(dim=1)[0]
                    confidence_scores.extend(max_probs.cpu().numpy())
                    
                    # Processing time
                    batch_time = time.time() - batch_start_time
                    processing_times.append(batch_time)
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
                
                # Periodic memory cleanup
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Compute final metrics
        num_batches = len(data_loader)
        results = {
            'accuracy': total_correct / max(total_samples, 1),
            'total_loss': total_losses['total_loss'] / max(num_batches, 1),
            'task_loss': total_losses['task_loss'] / max(num_batches, 1),
            'smmd_loss': total_losses['smmd_loss'] / max(num_batches, 1),
            'contrastive_loss': total_losses['contrastive_loss'] / max(num_batches, 1),
            'num_samples': total_samples
        }
        
        # Add detailed metrics
        if return_detailed_metrics:
            results.update({
                'confidence_mean': np.mean(confidence_scores) if confidence_scores else 0.0,
                'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
                'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0.0,
                'throughput': total_samples / sum(processing_times) if processing_times else 0.0
            })
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(predictions_list, labels_list, 'standard')
        
        return results
    
    def evaluate_early_exit(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_detailed_metrics: bool = True,
        save_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Evaluation with early-exit based on prediction confidence.
        
        This function implements dynamic early exit where samples can terminate
        processing at different depths based on prediction stability and confidence.
        This mimics efficient human decision-making and reduces computational cost.
        
        Early Exit Strategy:
        1. Track prediction confidence across G-Ladder layers
        2. Exit when confidence exceeds threshold and prediction is stable
        3. Different samples may use different computational paths
        4. Measure efficiency gains from early termination
        
        Args:
            data_loader: DataLoader containing evaluation data
            return_detailed_metrics: Whether to compute detailed metrics
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary containing evaluation metrics including efficiency measures
        """
        self.model.eval()
        
        # Early exit configuration
        confidence_threshold = self.config.inference.confidence_threshold
        min_layers = self.config.inference.min_exit_layer
        max_layers = self.config.g_ladder.num_layers
        
        # Initialize metrics
        total_correct = 0
        total_samples = 0
        total_losses = defaultdict(float)
        predictions_list = []
        labels_list = []
        
        # Early exit specific metrics
        exit_layer_counts = defaultdict(int)
        computational_costs = []
        
        # Detailed metrics
        if return_detailed_metrics:
            confidence_scores = []
            early_exit_rates = defaultdict(int)
            processing_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Early Exit Evaluation")):
                batch_start_time = time.time()
                
                # Prepare batch
                batch = self._prepare_batch(batch)
                batch_size = self._get_batch_size(batch)
                total_samples += batch_size
                
                # Initialize per-sample tracking
                sample_predictions = torch.full(
                    (batch_size,), -1, device=self.device, dtype=torch.long
                )
                sample_confidences = torch.zeros(
                    batch_size, device=self.device, dtype=torch.float
                )
                samples_active = torch.ones(
                    batch_size, device=self.device, dtype=torch.bool
                )
                sample_exit_layers = torch.full(
                    (batch_size,), max_layers, device=self.device, dtype=torch.long
                )
                
                # Progressive evaluation through G-Ladder layers
                current_text_embeddings = None
                
                for layer_idx in range(max_layers):
                    if not samples_active.any():
                        break  # All samples have exited
                    
                    # Forward pass through current layer
                    if layer_idx == 0:
                        # Initial processing
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            edge_index=batch['edge_index'],
                            node_features=batch.get('node_features'),
                            batch_nodes=batch.get('batch_nodes'),
                            anchor_mask=batch.get('anchor_mask'),
                            force_layers=layer_idx + 1  # Process only up to this layer
                        )
                    else:
                        # Continue from previous layer
                        outputs = self.model.forward_from_layer(
                            layer_idx=layer_idx,
                            previous_embeddings=current_text_embeddings,
                            edge_index=batch['edge_index'],
                            **batch
                        )
                    
                    current_text_embeddings = outputs.get('text_embeddings')
                    
                    # Get predictions and confidence for active samples
                    current_logits = outputs['logits'][samples_active]
                    current_probs = F.softmax(current_logits, dim=1)
                    current_preds = current_probs.argmax(dim=1)
                    current_conf = current_probs.max(dim=1)[0]
                    
                    # Early exit decision
                    if layer_idx >= min_layers:
                        # Check for early exit conditions
                        confident_mask = current_conf > confidence_threshold
                        
                        # Additional stability check: prediction hasn't changed
                        if layer_idx > min_layers:
                            previous_preds = sample_predictions[samples_active]
                            stable_mask = (current_preds == previous_preds) | (previous_preds == -1)
                            exit_mask = confident_mask & stable_mask
                        else:
                            exit_mask = confident_mask
                        
                        # Update exiting samples
                        if exit_mask.any():
                            # Get indices of samples to exit
                            active_indices = torch.where(samples_active)[0]
                            exit_indices = active_indices[exit_mask]
                            
                            # Update predictions and confidences
                            sample_predictions[exit_indices] = current_preds[exit_mask]
                            sample_confidences[exit_indices] = current_conf[exit_mask]
                            sample_exit_layers[exit_indices] = layer_idx + 1
                            
                            # Mark as inactive
                            samples_active[exit_indices] = False
                            
                            # Track exit statistics
                            exit_layer_counts[layer_idx + 1] += exit_mask.sum().item()
                    
                    # Update predictions for all active samples
                    if samples_active.any():
                        active_indices = torch.where(samples_active)[0]
                        sample_predictions[active_indices] = current_preds
                        sample_confidences[active_indices] = current_conf
                
                # Handle remaining active samples (use final layer predictions)
                if samples_active.any():
                    active_indices = torch.where(samples_active)[0]
                    sample_exit_layers[active_indices] = max_layers
                    exit_layer_counts[max_layers] += samples_active.sum().item()
                
                # Compute final losses using last layer outputs
                final_outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    edge_index=batch['edge_index'],
                    node_features=batch.get('node_features'),
                    batch_nodes=batch.get('batch_nodes'),
                    anchor_mask=batch.get('anchor_mask')
                )
                
                losses = self.model.compute_loss(
                    outputs=final_outputs,
                    labels=batch['labels'],
                    edge_index=batch['edge_index'],
                    source_data=batch.get('source_data'),
                    target_data=batch.get('target_data')
                )
                
                # Update metrics
                for loss_name, loss_value in losses.items():
                    total_losses[loss_name] += loss_value.item()
                
                correct = (sample_predictions == batch['labels']).sum().item()
                total_correct += correct
                
                # Computational cost tracking
                avg_layers_used = sample_exit_layers.float().mean().item()
                computational_costs.append(avg_layers_used)
                
                # Store predictions
                if save_predictions:
                    predictions_list.extend(sample_predictions.cpu().numpy())
                    labels_list.extend(batch['labels'].cpu().numpy())
                
                # Detailed metrics
                if return_detailed_metrics:
                    confidence_scores.extend(sample_confidences.cpu().numpy())
                    processing_times.append(time.time() - batch_start_time)
                    
                    # Track early exit rates
                    for layer, count in exit_layer_counts.items():
                        early_exit_rates[f'layer_{layer}'] = count
        
        # Compute final metrics
        num_batches = len(data_loader)
        results = {
            'accuracy': total_correct / max(total_samples, 1),
            'total_loss': total_losses['total_loss'] / max(num_batches, 1),
            'task_loss': total_losses['task_loss'] / max(num_batches, 1),
            'smmd_loss': total_losses['smmd_loss'] / max(num_batches, 1),
            'contrastive_loss': total_losses['contrastive_loss'] / max(num_batches, 1),
            'avg_computational_cost': np.mean(computational_costs) if computational_costs else max_layers,
            'computational_savings': 1.0 - (np.mean(computational_costs) / max_layers) if computational_costs else 0.0,
            'num_samples': total_samples
        }
        
        # Add early exit statistics
        results.update({
            f'exit_rate_layer_{layer}': count / max(total_samples, 1) * 100
            for layer, count in exit_layer_counts.items()
        })
        
        # Detailed metrics
        if return_detailed_metrics:
            results.update({
                'confidence_mean': np.mean(confidence_scores) if confidence_scores else 0.0,
                'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
                'throughput': total_samples / sum(processing_times) if processing_times else 0.0
            })
        
        # Save predictions
        if save_predictions:
            self._save_predictions(predictions_list, labels_list, 'early_exit')
        
        return results
    
    def evaluate_budget_aware(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_detailed_metrics: bool = True,
        save_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Budget-aware evaluation with selective LLM refresh on anchors.
        
        This function implements the ENGINE-GraphLoRA evaluation strategy with
        budget constraints on LLM refreshes. Only a subset of anchor nodes
        receive fresh LLM embeddings while others use cached representations.
        
        Budget Strategy:
        1. Select anchor nodes based on importance/centrality
        2. Allocate budget for LLM refreshes across batches
        3. Use cached embeddings for non-budget nodes
        4. Measure accuracy vs computational trade-offs
        
        Args:
            data_loader: DataLoader containing evaluation data
            return_detailed_metrics: Whether to compute detailed metrics
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary containing evaluation metrics with budget efficiency measures
        """
        self.model.eval()
        
        # Budget configuration
        budget_ratio = self.config.anchor.anchor_budget_ratio
        total_budget = int(len(data_loader.dataset) * budget_ratio)
        remaining_budget = total_budget
        
        logger.info(f"Budget-aware evaluation with budget ratio: {budget_ratio:.2f}")
        logger.info(f"Total LLM refresh budget: {total_budget} nodes")
        
        # Initialize metrics
        total_correct = 0
        total_samples = 0
        total_losses = defaultdict(float)
        predictions_list = []
        labels_list = []
        
        # Budget tracking
        budget_usage = []
        cache_hits = 0
        llm_refreshes = 0
        
        # Detailed metrics
        if return_detailed_metrics:
            confidence_scores = []
            processing_times = []
            anchor_accuracy = []
            non_anchor_accuracy = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Budget-Aware Evaluation")):
                batch_start_time = time.time()
                
                # Prepare batch
                batch = self._prepare_batch(batch)
                batch_size = self._get_batch_size(batch)
                total_samples += batch_size
                
                # Determine anchor nodes and budget allocation for this batch
                anchor_mask = batch.get('anchor_mask')
                if anchor_mask is None:
                    # Create anchor mask based on node importance
                    anchor_mask = self._create_anchor_mask(batch, batch_size)
                
                # Budget allocation for this batch
                anchor_count = anchor_mask.sum().item()
                batch_budget = min(anchor_count, remaining_budget)
                
                if batch_budget > 0:
                    # Select which anchors get LLM refresh
                    anchor_indices = torch.where(anchor_mask)[0]
                    if len(anchor_indices) > batch_budget:
                        # Prioritize anchors by importance
                        selected_indices = self._select_budget_anchors(
                            batch, anchor_indices, batch_budget
                        )
                        refresh_mask = torch.zeros_like(anchor_mask)
                        refresh_mask[selected_indices] = True
                    else:
                        refresh_mask = anchor_mask.clone()
                    
                    llm_refreshes += refresh_mask.sum().item()
                    remaining_budget -= refresh_mask.sum().item()
                else:
                    refresh_mask = torch.zeros_like(anchor_mask)
                
                cache_hits += (anchor_mask & ~refresh_mask).sum().item()
                
                # Forward pass with budget constraints
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    edge_index=batch['edge_index'],
                    node_features=batch.get('node_features'),
                    batch_nodes=batch.get('batch_nodes'),
                    anchor_mask=anchor_mask,
                    refresh_mask=refresh_mask  # Only refresh selected anchors
                )
                
                # Compute losses
                losses = self.model.compute_loss(
                    outputs=outputs,
                    labels=batch['labels'],
                    edge_index=batch['edge_index'],
                    source_data=batch.get('source_data'),
                    target_data=batch.get('target_data')
                )
                
                # Update metrics
                for loss_name, loss_value in losses.items():
                    total_losses[loss_name] += loss_value.item()
                
                predictions = outputs['logits'].argmax(dim=1)
                correct = (predictions == batch['labels']).sum().item()
                total_correct += correct
                
                # Track budget usage
                budget_used = refresh_mask.sum().item()
                budget_usage.append(budget_used)
                
                # Store predictions
                if save_predictions:
                    predictions_list.extend(predictions.cpu().numpy())
                    labels_list.extend(batch['labels'].cpu().numpy())
                
                # Detailed metrics
                if return_detailed_metrics:
                    # Overall confidence
                    probs = F.softmax(outputs['logits'], dim=1)
                    confidence_scores.extend(probs.max(dim=1)[0].cpu().numpy())
                    
                    # Separate accuracy for anchor vs non-anchor nodes
                    if anchor_mask.any():
                        anchor_correct = (predictions[anchor_mask] == batch['labels'][anchor_mask]).float().mean().item()
                        anchor_accuracy.append(anchor_correct)
                    
                    if (~anchor_mask).any():
                        non_anchor_correct = (predictions[~anchor_mask] == batch['labels'][~anchor_mask]).float().mean().item()
                        non_anchor_accuracy.append(non_anchor_correct)
                    
                    processing_times.append(time.time() - batch_start_time)
        
        # Compute final metrics
        num_batches = len(data_loader)
        total_anchors = cache_hits + llm_refreshes
        
        results = {
            'accuracy': total_correct / max(total_samples, 1),
            'total_loss': total_losses['total_loss'] / max(num_batches, 1),
            'task_loss': total_losses['task_loss'] / max(num_batches, 1),
            'smmd_loss': total_losses['smmd_loss'] / max(num_batches, 1),
            'contrastive_loss': total_losses['contrastive_loss'] / max(num_batches, 1),
            'budget_usage': sum(budget_usage),
            'budget_efficiency': total_correct / max(sum(budget_usage), 1),  # Accuracy per LLM refresh
            'cache_hit_rate': cache_hits / max(total_anchors, 1) * 100,
            'llm_refresh_rate': llm_refreshes / max(total_anchors, 1) * 100,
            'num_samples': total_samples
        }
        
        # Detailed metrics
        if return_detailed_metrics:
            results.update({
                'confidence_mean': np.mean(confidence_scores) if confidence_scores else 0.0,
                'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0,
                'anchor_accuracy': np.mean(anchor_accuracy) if anchor_accuracy else 0.0,
                'non_anchor_accuracy': np.mean(non_anchor_accuracy) if non_anchor_accuracy else 0.0,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
                'throughput': total_samples / sum(processing_times) if processing_times else 0.0
            })
        
        # Save predictions
        if save_predictions:
            self._save_predictions(predictions_list, labels_list, 'budget_aware')
        
        logger.info(f"Budget utilization: {sum(budget_usage)}/{total_budget} ({sum(budget_usage)/max(total_budget,1)*100:.1f}%)")
        logger.info(f"Cache hit rate: {results['cache_hit_rate']:.1f}%")
        
        return results
    
    def evaluate_efficiency_analysis(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """
        Comprehensive efficiency analysis across all evaluation modes.
        
        Args:
            data_loader: DataLoader containing evaluation data
            
        Returns:
            Dictionary containing efficiency analysis results
        """
        logger.info("Starting comprehensive efficiency analysis...")
        
        # Run all evaluation modes
        results = self.comprehensive_evaluate(
            data_loader,
            evaluation_modes=['standard', 'early_exit', 'budget_aware'],
            return_detailed_metrics=True
        )
        
        # Compute efficiency metrics
        efficiency_analysis = {
            'accuracy_comparison': {
                mode: results[mode]['accuracy'] for mode in results
            },
            'computational_cost_comparison': {
                'standard': results['standard'].get('avg_memory_usage', 0.0),
                'early_exit': results['early_exit'].get('avg_computational_cost', 0.0),
                'budget_aware': results['budget_aware'].get('budget_usage', 0.0)
            },
            'throughput_comparison': {
                mode: results[mode].get('throughput', 0.0) for mode in results
            },
            'efficiency_ratios': {
                'early_exit_speedup': (
                    results['early_exit'].get('throughput', 0.0) / 
                    max(results['standard'].get('throughput', 1.0), 1.0)
                ),
                'budget_efficiency': results['budget_aware'].get('budget_efficiency', 0.0),
                'computational_savings': results['early_exit'].get('computational_savings', 0.0)
            }
        }
        
        return efficiency_analysis
    
    # Helper methods
    
    def _prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Prepare batch data for evaluation"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        else:
            # Handle torch_geometric Data objects
            batch = batch.to(self.device)
            return {
                'input_ids': getattr(batch, 'input_ids', None),
                'attention_mask': getattr(batch, 'attention_mask', None),
                'edge_index': batch.edge_index,
                'node_features': getattr(batch, 'x', None),
                'labels': batch.y,
                'batch_nodes': getattr(batch, 'batch', None),
                'anchor_mask': getattr(batch, 'anchor_mask', None)
            }
    
    def _get_batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """Get batch size from batch data"""
        if 'labels' in batch:
            return len(batch['labels'])
        elif 'input_ids' in batch:
            return len(batch['input_ids'])
        else:
            return 1
    
    def _create_anchor_mask(self, batch: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Create anchor mask based on node importance"""
        # Simple implementation: use high-degree nodes as anchors
        # In practice, this should use pre-computed centrality scores
        anchor_ratio = self.config.anchor.anchor_ratio
        num_anchors = max(1, int(batch_size * anchor_ratio))
        
        # Create random anchor mask for demonstration
        anchor_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        anchor_indices = torch.randperm(batch_size)[:num_anchors]
        anchor_mask[anchor_indices] = True
        
        return anchor_mask
    
    def _select_budget_anchors(
        self,
        batch: Dict[str, torch.Tensor],
        anchor_indices: torch.Tensor,
        budget: int
    ) -> torch.Tensor:
        """Select which anchors get LLM refresh based on importance"""
        # Simple implementation: select randomly
        # In practice, this should prioritize based on centrality, uncertainty, etc.
        selected_count = min(len(anchor_indices), budget)
        perm = torch.randperm(len(anchor_indices))[:selected_count]
        return anchor_indices[perm]
    
    def _save_predictions(self, predictions: List, labels: List, mode: str):
        """Save predictions and labels to file"""
        output_dir = Path(self.config.output_dir) / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'predictions_{mode}.npz'
        np.savez(output_file, predictions=predictions, labels=labels)
        logger.info(f"Saved predictions to {output_file}")


def evaluate_model(
    model: EngineGraphLoRAModel,
    data_loader: torch.utils.data.DataLoader,
    config: EngineGraphLoRAConfig,
    evaluation_modes: List[str] = ['standard'],
    return_detailed_metrics: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Convenient function for model evaluation.
    
    Args:
        model: ENGINE-GraphLoRA model to evaluate
        data_loader: DataLoader containing evaluation data
        config: Configuration object
        evaluation_modes: List of evaluation modes to run
        return_detailed_metrics: Whether to compute detailed metrics
        
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = EngineGraphLoRAEvaluator(model, config)
    return evaluator.comprehensive_evaluate(
        data_loader, evaluation_modes, return_detailed_metrics
    )


def main():
    """Example usage of the evaluation module"""
    
    # Parse arguments
    args = EngineGraphLoRAArguments()
    config = args.parse_args()
    
    # Initialize model (placeholder)
    model = EngineGraphLoRAModel(config, num_classes=10)
    
    # Initialize evaluator
    evaluator = EngineGraphLoRAEvaluator(model, config)
    
    print("ENGINE-GraphLoRA Evaluator initialized successfully!")
    print(f"Evaluation modes available: standard, early_exit, budget_aware")


if __name__ == "__main__":
    main() 