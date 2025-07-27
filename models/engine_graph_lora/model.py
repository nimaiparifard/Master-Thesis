"""
ENGINE-GraphLoRA Model: Unified Text-Graph Learning Framework

This module implements the complete ENGINE-GraphLoRA model that combines:
- ENGINE approach: G-Ladder modules for text-graph fusion
- GraphLoRA approach: Parameter-efficient adaptation with LoRA
- Anchor-based selective LLM refresh
- Multi-objective optimization with SMMD, contrastive, and structural losses

Author: Graph Neural Network Research Team
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import AutoModel, AutoTokenizer
import logging

from .config import EngineGraphLoRAConfig
from .g_ladder import GLadderLM
from .lora_sage import LoRASAGE, LoRAGAT
from .anchor_system import AnchorSelector, CacheManager, PromptRouter
from .losses import StructureAwareMMD, ContrastiveTextStructureLoss
from .cross_modal import CrossModalFusion, EarlyExitModule

logger = logging.getLogger(__name__)


class EngineGraphLoRAModel(nn.Module):
    """
    Main ENGINE-GraphLoRA model that combines text-graph fusion with parameter-efficient adaptation.
    
    Architecture:
    1. Frozen LLM backbone with cached embeddings
    2. G-Ladder modules for cross-layer graph-text fusion  
    3. LoRA-enhanced GNN for parameter-efficient adaptation
    4. Anchor-based selective refresh mechanism
    5. Cross-modal fusion and early-exit inference
    """
    
    def __init__(self, config: EngineGraphLoRAConfig, num_classes: int, vocab_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Initialize components
        self._init_llm_backbone()
        self._init_g_ladder()
        self._init_gnn_layers() 
        self._init_anchor_system()
        self._init_cross_modal_fusion()
        self._init_prediction_head()
        self._init_loss_functions()
        
        # Training state
        self.training_step = 0
        self.current_epoch = 0
        
        logger.info(f"Initialized ENGINE-GraphLoRA model with {self.count_parameters():,} parameters")
    
    def _get_config_value(self, section: str, key: str, default=None):
        """Safely get configuration value from nested config (handles both dict and object)"""
        config_section = getattr(self.config, section, {})
        if isinstance(config_section, dict):
            return config_section.get(key, default)
        else:
            return getattr(config_section, key, default)
        
    def _init_llm_backbone(self):
        """Initialize the frozen LLM backbone"""
        model_name = self._get_config_value('llm', 'model_name', 'prajjwal1/bert-tiny')
        use_fp16 = self._get_config_value('training', 'use_fp16', False)
        freeze_base = self._get_config_value('llm', 'freeze_base', True)
        gradient_checkpointing = self._get_config_value('llm', 'gradient_checkpointing', False)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        
        # Freeze base LLM parameters
        if freeze_base:
            for param in self.llm.parameters():
                param.requires_grad = False
                
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
            
    def _init_g_ladder(self):
        """Initialize G-Ladder modules for each LLM layer"""
        self.g_ladder = GLadderLM(
            hidden_size=self._get_config_value('llm', 'hidden_size', 128),
            num_layers=self._get_config_value('llm', 'num_layers', 2),
            lora_rank=self._get_config_value('g_ladder', 'lora_rank', 16),
            lora_alpha=self._get_config_value('g_ladder', 'lora_alpha', 32.0),
            lora_dropout=self._get_config_value('g_ladder', 'lora_dropout', 0.1),
            mp_type=self._get_config_value('g_ladder', 'mp_type', 'mean'),
            use_gating=self._get_config_value('g_ladder', 'use_gating', True),
            gate_activation=self._get_config_value('g_ladder', 'gate_activation', 'sigmoid'),
            injection_layers=self._get_config_value('g_ladder', 'injection_layers', [0, 1]),
            weight_sharing=self._get_config_value('g_ladder', 'weight_sharing', False)
        )
        
    def _init_gnn_layers(self):
        """Initialize LoRA-enhanced GNN layers"""
        gnn_type = self._get_config_value('gnn', 'type', 'sage')
        
        if gnn_type.lower() == 'sage':
            self.gnn = LoRASAGE(
                input_size=self._get_config_value('llm', 'hidden_size', 128),
                hidden_size=self._get_config_value('gnn', 'hidden_size', 64),
                output_size=self._get_config_value('gnn', 'hidden_size', 64),  # Use hidden_size as output_size
                num_layers=self._get_config_value('gnn', 'num_layers', 2),
                lora_rank=self._get_config_value('lora', 'rank', 4),
                lora_alpha=self._get_config_value('lora', 'alpha', 8.0),
                lora_dropout=self._get_config_value('lora', 'dropout', 0.1),
                dropout=self._get_config_value('gnn', 'dropout', 0.1),
                activation=self._get_config_value('gnn', 'activation', 'relu'),
                aggregator=self._get_config_value('gnn', 'aggregator', 'mean'),
                use_residual=self._get_config_value('gnn', 'use_residual', True)
            )
        elif gnn_type.lower() == 'gat':
            self.gnn = LoRAGAT(
                input_size=self._get_config_value('llm', 'hidden_size', 128),
                hidden_size=self._get_config_value('gnn', 'hidden_size', 64),
                output_size=self._get_config_value('gnn', 'hidden_size', 64),
                num_layers=self._get_config_value('gnn', 'num_layers', 2),
                num_heads=self._get_config_value('gnn', 'num_heads', 4),
                lora_rank=self._get_config_value('lora', 'rank', 4),
                lora_alpha=self._get_config_value('lora', 'alpha', 8.0),
                lora_dropout=self._get_config_value('lora', 'dropout', 0.1),
                dropout=self._get_config_value('gnn', 'dropout', 0.1),
                head_dropout=self._get_config_value('gnn', 'head_dropout', 0.1),
                use_residual=self._get_config_value('gnn', 'use_residual', True)
            )
        else:
            print(f"Warning: Unsupported GNN type '{gnn_type}', using SAGE as fallback")
            self.gnn = LoRASAGE(
                input_size=self._get_config_value('llm', 'hidden_size', 128),
                hidden_size=self._get_config_value('gnn', 'hidden_size', 64),
                output_size=self._get_config_value('gnn', 'hidden_size', 64),
                num_layers=self._get_config_value('gnn', 'num_layers', 2),
                lora_rank=self._get_config_value('lora', 'rank', 4),
                lora_alpha=self._get_config_value('lora', 'alpha', 8.0),
                lora_dropout=self._get_config_value('lora', 'dropout', 0.1),
                dropout=self._get_config_value('gnn', 'dropout', 0.1),
                activation=self._get_config_value('gnn', 'activation', 'relu'),
                aggregator=self._get_config_value('gnn', 'aggregator', 'mean'),
                use_residual=self._get_config_value('gnn', 'use_residual', True)
            )
            
    def _init_anchor_system(self):
        """Initialize anchor selection and caching system"""
        try:
            self.anchor_selector = AnchorSelector(
                anchor_ratio=self._get_config_value('anchor', 'anchor_ratio', 0.2),
                selection_method=self._get_config_value('anchor', 'selection_method', 'degree'),
                pagerank_weight=self._get_config_value('anchor', 'pagerank_weight', 0.5),
                degree_weight=self._get_config_value('anchor', 'degree_weight', 0.3),
                entropy_weight=self._get_config_value('anchor', 'entropy_weight', 0.2)
            )
        except Exception as e:
            print(f"Warning: Could not initialize anchor selector: {e}")
            self.anchor_selector = None
        
        try:
            self.cache_manager = CacheManager(
                cache_dir=self._get_config_value('llm', 'cache_dir', './cache'),
                cache_format=self._get_config_value('llm', 'cache_format', 'pt'),
                hidden_size=self._get_config_value('llm', 'hidden_size', 128)
            )
        except Exception as e:
            print(f"Warning: Could not initialize cache manager: {e}")
            self.cache_manager = None
        
        try:
            self.prompt_router = PromptRouter(
                hidden_size=self._get_config_value('anchor', 'router_hidden_size', 128),
                num_layers=self._get_config_value('anchor', 'router_layers', 2),
                temperature=self._get_config_value('anchor', 'router_temperature', 1.0),
                threshold=self._get_config_value('anchor', 'router_threshold', 0.5)
            )
        except Exception as e:
            print(f"Warning: Could not initialize prompt router: {e}")
            self.prompt_router = None
        
    def _init_cross_modal_fusion(self):
        """Initialize cross-modal fusion modules"""
        try:
            fusion_hidden_size = self._get_config_value('cross_modal', 'fusion_hidden_size', 128)
            
            self.cross_modal_fusion = CrossModalFusion(
                text_hidden_size=self._get_config_value('llm', 'hidden_size', 128),
                graph_hidden_size=self._get_config_value('gnn', 'hidden_size', 64),
                fusion_hidden_size=fusion_hidden_size,
                fusion_method=self._get_config_value('cross_modal', 'fusion_method', 'concat'),
                num_layers=self._get_config_value('cross_modal', 'fusion_layers', 2),
                num_heads=self._get_config_value('cross_modal', 'num_attention_heads', 4),
                dropout=self._get_config_value('cross_modal', 'attention_dropout', 0.1)
            )
        except Exception as e:
            print(f"Warning: Could not initialize cross-modal fusion, using simple concatenation: {e}")
            # Simple fallback
            text_size = self._get_config_value('llm', 'hidden_size', 128)
            graph_size = self._get_config_value('gnn', 'hidden_size', 64)
            self.cross_modal_fusion = nn.Linear(text_size + graph_size, text_size + graph_size)
            
        try:
            if self._get_config_value('cross_modal', 'enable_early_exit', False):
                self.early_exit = EarlyExitModule(
                    hidden_size=self._get_config_value('cross_modal', 'fusion_hidden_size', 128),
                    threshold=self._get_config_value('cross_modal', 'exit_threshold', 0.8),
                    min_layers=self._get_config_value('cross_modal', 'min_layers', 1),
                    max_layers=self._get_config_value('cross_modal', 'max_layers', 3)
                )
            else:
                self.early_exit = None
        except Exception as e:
            print(f"Warning: Could not initialize early exit module: {e}")
            self.early_exit = None
            
    def _init_prediction_head(self):
        """Initialize prediction head for downstream tasks"""
        fusion_hidden_size = self._get_config_value('cross_modal', 'fusion_hidden_size', 128)
        
        # If cross-modal fusion failed, use combined text+graph size
        if hasattr(self.cross_modal_fusion, 'in_features'):
            # Fallback linear layer was used
            fusion_hidden_size = self.cross_modal_fusion.in_features
        
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_size // 2, self.num_classes)
        )
        
    def _init_loss_functions(self):
        """Initialize loss functions"""
        self.smmd_loss = StructureAwareMMD(
            kernel=self.config.loss.smmd_kernel,
            bandwidth=self.config.loss.smmd_bandwidth,
            include_degree=self.config.loss.smmd_include_degree,
            include_clustering=self.config.loss.smmd_include_clustering
        )
        
        self.contrastive_loss = ContrastiveTextStructureLoss(
            temperature=self.config.loss.contrastive_temperature,
            negative_sampling_ratio=self.config.loss.negative_sampling_ratio,
            hard_negative_mining=self.config.loss.hard_negative_mining
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        edge_index: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        batch_nodes: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing the ENGINE-GraphLoRA workflow
        
        Args:
            input_ids: Tokenized text inputs [num_nodes, seq_len]
            attention_mask: Attention mask for text [num_nodes, seq_len]
            edge_index: Graph edges [2, num_edges]
            node_features: Optional node features [num_nodes, feat_dim]
            batch_nodes: Batch node indices [batch_size]
            anchor_mask: Mask indicating anchor nodes [num_nodes]
            
        Returns:
            Dictionary containing model outputs and intermediate states
        """
        batch_size = input_ids.size(0) if batch_nodes is None else len(batch_nodes)
        device = input_ids.device
        
        outputs = {}
        
        # ===== STEP 1: Text Embedding (with caching and selective refresh) =====
        h_text = self._process_text_embeddings(
            input_ids, attention_mask, batch_nodes, anchor_mask
        )
        outputs['text_embeddings'] = h_text
        
        # ===== STEP 2: G-Ladder Cross-Layer Fusion =====
        h_fused = self._apply_g_ladder_fusion(h_text, edge_index, batch_nodes)
        outputs['fused_embeddings'] = h_fused
        
        # ===== STEP 3: LoRA-Enhanced GNN Message Passing =====
        z_graph = self._apply_gnn_layers(h_fused.detach(), edge_index, batch_nodes)
        outputs['graph_embeddings'] = z_graph
        
        # ===== STEP 4: Cross-Modal Fusion =====
        c_joint = self.cross_modal_fusion(h_fused, z_graph)
        outputs['joint_embeddings'] = c_joint
        
        # ===== STEP 5: Early Exit (if enabled) =====
        if self.early_exit is not None and not self.training:
            c_joint, exit_layer = self.early_exit(c_joint)
            outputs['exit_layer'] = exit_layer
            
        # ===== STEP 6: Prediction =====
        logits = self.prediction_head(c_joint)
        outputs['logits'] = logits
        
        # Store intermediate states for loss computation
        if self.training:
            outputs.update({
                'text_hidden_states': h_text,
                'graph_hidden_states': z_graph,
                'anchor_mask': anchor_mask
            })
            
        return outputs
    
    def _process_text_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_nodes: Optional[torch.Tensor],
        anchor_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Process text embeddings with caching and selective refresh"""
        device = input_ids.device
        num_nodes = input_ids.size(0)
        
        # Initialize embeddings from cache
        if self.config.llm.cache_embeddings and not self.training:
            h_cached = self.cache_manager.get_embeddings(batch_nodes)
            if h_cached is not None:
                return h_cached.to(device)
        
        # Determine which nodes need LLM refresh
        if self.training and anchor_mask is not None:
            # During training: use prompt router + anchor mask
            refresh_scores = self.prompt_router(input_ids, attention_mask)
            
            if self.config.anchor.refresh_strategy == "stochastic":
                refresh_probs = torch.sigmoid(refresh_scores / self.config.anchor.router_temperature)
                refresh_mask = (torch.rand_like(refresh_probs) < refresh_probs) & anchor_mask.bool()
            else:
                refresh_mask = (refresh_scores > self.config.anchor.router_threshold) & anchor_mask.bool()
        else:
            # During inference or no anchor mask: refresh all
            refresh_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        
        # Get cached embeddings for non-refresh nodes
        h_text = torch.zeros(num_nodes, self.config.llm.hidden_size, device=device)
        
        if not refresh_mask.all():
            non_refresh_nodes = (~refresh_mask).nonzero().squeeze(-1)
            if len(non_refresh_nodes) > 0:
                h_cached = self.cache_manager.get_embeddings(non_refresh_nodes)
                if h_cached is not None:
                    h_text[non_refresh_nodes] = h_cached.to(device)
        
        # Process refreshed nodes through LLM
        if refresh_mask.any():
            refresh_indices = refresh_mask.nonzero().squeeze(-1)
            refresh_input_ids = input_ids[refresh_indices]
            refresh_attention_mask = attention_mask[refresh_indices]
            
            with torch.cuda.amp.autocast(enabled=self.config.training.use_amp):
                llm_outputs = self.llm(
                    input_ids=refresh_input_ids,
                    attention_mask=refresh_attention_mask,
                    output_hidden_states=True
                )
                
            # Extract final hidden states (mean pooling over sequence)
            h_refresh = llm_outputs.last_hidden_state
            h_refresh = (h_refresh * refresh_attention_mask.unsqueeze(-1)).sum(dim=1) / refresh_attention_mask.sum(dim=1, keepdim=True)
            
            h_text[refresh_indices] = h_refresh
            
            # Update cache
            if self.config.llm.cache_embeddings:
                self.cache_manager.update_embeddings(refresh_indices, h_refresh.detach().cpu())
        
        return h_text
    
    def _apply_g_ladder_fusion(
        self,
        h_text: torch.Tensor,
        edge_index: torch.Tensor,
        batch_nodes: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply G-Ladder modules for cross-layer graph-text fusion"""
        # Prepare adjacency for message passing
        num_nodes = h_text.size(0)
        
        # Apply G-Ladder fusion
        h_fused = self.g_ladder(h_text, edge_index, num_nodes)
        
        return h_fused
    
    def _apply_gnn_layers(
        self,
        h_input: torch.Tensor,
        edge_index: torch.Tensor,
        batch_nodes: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply LoRA-enhanced GNN layers"""
        z_graph = self.gnn(h_input, edge_index)
        return z_graph
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        source_data: Optional[Dict] = None,
        target_data: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss combining task, SMMD, contrastive, and structural components
        
        Args:
            outputs: Model outputs from forward pass
            labels: Ground truth labels
            edge_index: Graph edges
            source_data: Source domain data for SMMD
            target_data: Target domain data for SMMD
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # ===== Task Loss (Classification) =====
        logits = outputs['logits']
        task_loss = F.cross_entropy(logits, labels)
        losses['task_loss'] = task_loss
        
        # ===== Structure-Aware MMD Loss =====
        if source_data is not None and target_data is not None:
            smmd_loss = self.smmd_loss(
                source_embeddings=source_data.get('embeddings', outputs['graph_embeddings']),
                target_embeddings=outputs['graph_embeddings'],
                source_structure=source_data.get('structure_features'),
                target_structure=target_data.get('structure_features')
            )
            losses['smmd_loss'] = smmd_loss
        else:
            losses['smmd_loss'] = torch.tensor(0.0, device=logits.device)
        
        # ===== Contrastive Text-Structure Loss =====
        if 'text_hidden_states' in outputs and 'graph_hidden_states' in outputs:
            contrastive_loss = self.contrastive_loss(
                text_embeddings=outputs['text_hidden_states'],
                graph_embeddings=outputs['graph_hidden_states'],
                edge_index=edge_index,
                labels=labels
            )
            losses['contrastive_loss'] = contrastive_loss
        else:
            losses['contrastive_loss'] = torch.tensor(0.0, device=logits.device)
        
        # ===== Structure Regularization Loss =====
        structure_loss = self._compute_structure_regularization(
            outputs['graph_embeddings'], edge_index, labels
        )
        losses['structure_loss'] = structure_loss
        
        # ===== Combined Loss =====
        total_loss = (
            self.config.loss.task_weight * losses['task_loss'] +
            self.config.loss.smmd_weight * losses['smmd_loss'] +
            self.config.loss.contrastive_weight * losses['contrastive_loss'] +
            self.config.loss.structure_weight * losses['structure_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_structure_regularization(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute homophily-based structure regularization loss"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Get edge embeddings
        src_embeddings = embeddings[edge_index[0]]
        dst_embeddings = embeddings[edge_index[1]]
        
        # Compute similarity
        similarities = F.cosine_similarity(src_embeddings, dst_embeddings, dim=1)
        
        # Get edge labels
        src_labels = labels[edge_index[0]]
        dst_labels = labels[edge_index[1]]
        same_label = (src_labels == dst_labels).float()
        
        # Homophily loss: encourage high similarity for same-label edges
        homophily_loss = F.mse_loss(similarities, same_label)
        
        return self.config.loss.homophily_lambda * homophily_loss
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> List[str]:
        """Get names of trainable parameters"""
        return [name for name, param in self.named_parameters() if param.requires_grad]
    
    def setup_for_training(self):
        """Setup model for training mode"""
        self.train()
        self.training_step = 0
        
        # Ensure LLM stays frozen
        if self.config.llm.freeze_base:
            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False
    
    def setup_for_inference(self):
        """Setup model for inference mode"""
        self.eval()
        
        # Ensure all modules are in eval mode
        for module in self.modules():
            module.eval()
    
    def forward_from_layer(
        self,
        layer_idx: int,
        previous_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass starting from a specific G-Ladder layer.
        Used for early-exit evaluation.
        """
        h_current = previous_embeddings
        
        # Apply G-Ladder fusion (simplified for compatibility)
        h_current = self.g_ladder(h_current, edge_index)
        
        # Apply GNN and get predictions
        z_graph = self.gnn(h_current, edge_index)
        logits = self.prediction_head(z_graph)
        
        return {
            'logits': logits,
            'text_embeddings': h_current,
            'graph_embeddings': z_graph
        } 