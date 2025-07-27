"""
ENGINE-GraphLoRA: Integrating ENGINE and GraphLoRA Approaches

This module combines the ENGINE approach (G-Ladder modules for text-graph fusion)
with GraphLoRA's parameter-efficient adaptation techniques to create a unified
framework for text-attributed graph learning.

Key Components:
- G-Ladder modules for cross-layer graph-text fusion
- LoRA-enhanced GNN layers for parameter-efficient adaptation  
- Structure-aware MMD loss for domain alignment
- Contrastive text-structure learning
- Anchor-based selective LLM refresh mechanism
- Cross-modal fusion and early-exit inference

Author: Graph Neural Network Research Team
Version: 1.0
"""

from .g_ladder import GLadderModule, GLadderLM
from .lora_sage import LoRASAGE, LoRAGAT
from .anchor_system import AnchorSelector, CacheManager, PromptRouter
from .losses import StructureAwareMMD, ContrastiveTextStructureLoss
from .cross_modal import CrossModalFusion, EarlyExitModule
from .trainer import EngineGraphLoRATrainer
from .config import EngineGraphLoRAConfig
from .model import EngineGraphLoRAModel

__all__ = [
    # Core modules
    "GLadderModule", 
    "GLadderLM",
    "LoRASAGE", 
    "LoRAGAT",
    
    # Anchor and caching system
    "AnchorSelector", 
    "CacheManager", 
    "PromptRouter",
    
    # Loss functions
    "StructureAwareMMD", 
    "ContrastiveTextStructureLoss",
    
    # Fusion and inference
    "CrossModalFusion", 
    "EarlyExitModule",
    
    # Training and configuration
    "EngineGraphLoRATrainer",
    "EngineGraphLoRAConfig",
    "EngineGraphLoRAModel",
] 