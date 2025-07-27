"""
ENGINE-GraphLoRA Configuration Module

This module provides comprehensive configuration management for the ENGINE-GraphLoRA
combined approach, integrating text-graph fusion with parameter-efficient adaptation.

Author: Graph Neural Network Research Team
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import torch


@dataclass
class LLMConfig:
    """Configuration for Language Model components"""
    # Model specifications
    model_name: str = "microsoft/DialoGPT-medium"  # Base LLM model
    model_size: str = "medium"  # small, medium, large
    hidden_size: int = 1024
    num_layers: int = 24
    vocab_size: int = 50257
    max_sequence_length: int = 512
    
    # Freezing and gradient control
    freeze_base: bool = True
    gradient_checkpointing: bool = True
    
    # Cache configuration
    cache_embeddings: bool = True
    cache_dir: str = "./cache/embeddings"
    cache_format: str = "h5"  # h5, torch, numpy


@dataclass
class GLadderConfig:
    """Configuration for G-Ladder modules (ENGINE approach)"""
    # Architecture
    num_layers: int = 24  # Should match LLM layers
    hidden_size: int = 1024
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    
    # Message passing configuration
    mp_type: str = "mean"  # mean, max, sum, attention
    use_gating: bool = True
    gate_activation: str = "sigmoid"
    
    # Layer-specific settings
    enable_per_layer: List[bool] = field(default_factory=lambda: [True] * 24)
    weight_sharing: bool = False  # Share weights across layers
    
    # Injection points (which LLM layers to inject into)
    injection_layers: List[int] = field(default_factory=lambda: list(range(0, 24, 2)))


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    # LoRA parameters
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    
    # Target modules for LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # LoRA-SAGE specific
    sage_layers: int = 3
    sage_hidden_size: int = 256
    sage_dropout: float = 0.1
    sage_activation: str = "relu"


@dataclass
class AnchorConfig:
    """Configuration for anchor node selection and caching"""
    # Anchor selection
    anchor_ratio: float = 0.15  # Percentage of nodes to select as anchors
    selection_method: str = "hybrid"  # pagerank, degree, entropy, hybrid
    
    # Centrality weights for hybrid selection
    pagerank_weight: float = 0.5
    degree_weight: float = 0.3
    entropy_weight: float = 0.2
    
    # Refresh strategy
    refresh_strategy: str = "stochastic"  # deterministic, stochastic, budget
    refresh_probability: float = 0.3
    budget_flops: int = 1000000  # FLOP budget per epoch
    
    # Router configuration
    router_hidden_size: int = 128
    router_layers: int = 2
    router_temperature: float = 0.5
    router_threshold: float = 0.7


@dataclass
class GNNConfig:
    """Configuration for Graph Neural Network components"""
    # Architecture
    gnn_type: str = "sage"  # sage, gat, gcn, gin
    num_layers: int = 3
    hidden_size: int = 256
    output_size: int = 256
    
    # Layer-specific settings
    dropout: float = 0.1
    activation: str = "relu"
    normalization: str = "batch"  # batch, layer, none
    
    # GAT-specific
    num_heads: int = 8
    head_dropout: float = 0.1
    
    # SAGE-specific
    aggregator: str = "mean"  # mean, max, lstm
    
    # Skip connections and residuals
    use_residual: bool = True
    use_skip_connections: bool = True


@dataclass
class LossConfig:
    """Configuration for loss functions and weights"""
    # Loss weights
    task_weight: float = 1.0
    smmd_weight: float = 0.1
    contrastive_weight: float = 0.05
    structure_weight: float = 0.01
    
    # Structure-aware MMD
    smmd_kernel: str = "rbf"  # rbf, linear, polynomial
    smmd_bandwidth: float = 1.0
    smmd_include_degree: bool = True
    smmd_include_clustering: bool = True
    
    # Contrastive learning
    contrastive_temperature: float = 0.1
    negative_sampling_ratio: int = 5
    hard_negative_mining: bool = True
    
    # Structure regularization
    homophily_lambda: float = 0.01
    structure_preservation_lambda: float = 0.001


@dataclass
class CrossModalConfig:
    """Configuration for cross-modal fusion"""
    # Fusion strategy
    fusion_method: str = "biaffine"  # biaffine, attention, concat, gate
    fusion_hidden_size: int = 512
    fusion_layers: int = 2
    
    # Injection points in LLM
    early_fusion_layer: int = 8   # L/3
    late_fusion_layer: int = 16   # 2L/3
    
    # Attention-based fusion
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Early exit configuration
    enable_early_exit: bool = True
    exit_threshold: float = 0.1  # Entropy threshold
    min_layers: int = 6  # Minimum layers before exit
    max_layers: int = 20  # Maximum layers


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Basic training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # linear, cosine, exponential
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd
    gradient_clip_norm: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Validation and checkpointing
    validation_frequency: int = 5
    save_frequency: int = 10
    early_stopping_patience: int = 20
    
    # Mixed precision
    use_fp16: bool = True
    use_amp: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Dataset specifications
    dataset_name: str = "cora"
    data_dir: str = "./data"
    
    # Text processing
    max_text_length: int = 512
    tokenizer_name: str = "microsoft/DialoGPT-medium"
    
    # Graph processing
    add_self_loops: bool = True
    normalize_features: bool = True
    
    # Data splits
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Few-shot settings
    few_shot_k: int = 5
    few_shot_episodes: int = 100


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    # Batch processing
    inference_batch_size: int = 64
    
    # Early exit
    use_early_exit: bool = True
    exit_confidence_threshold: float = 0.9
    
    # Caching
    use_cache: bool = True
    cache_non_anchors: bool = True
    
    # Output
    return_attention_weights: bool = False
    return_intermediate_outputs: bool = False


@dataclass 
class EngineGraphLoRAConfig:
    """Main configuration class combining all components"""
    
    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    g_ladder: GLadderConfig = field(default_factory=GLadderConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cross_modal: CrossModalConfig = field(default_factory=CrossModalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Global settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    debug: bool = False
    log_level: str = "INFO"
    
    # Experiment tracking
    experiment_name: str = "engine_graph_lora"
    run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and adjustments"""
        # Ensure consistency between components
        if self.g_ladder.hidden_size != self.llm.hidden_size:
            self.g_ladder.hidden_size = self.llm.hidden_size
            
        if self.g_ladder.num_layers != self.llm.num_layers:
            self.g_ladder.num_layers = self.llm.num_layers
            
        # Validate injection layers
        max_layer = self.llm.num_layers - 1
        self.g_ladder.injection_layers = [
            l for l in self.g_ladder.injection_layers if l <= max_layer
        ]
        
        # Set fusion layer bounds
        self.cross_modal.early_fusion_layer = min(
            self.cross_modal.early_fusion_layer, max_layer
        )
        self.cross_modal.late_fusion_layer = min(
            self.cross_modal.late_fusion_layer, max_layer
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EngineGraphLoRAConfig':
        """Create config from dictionary"""
        config = cls()
        
        for section_name, section_config in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_config, dict):
                section_obj = getattr(config, section_name)
                for key, value in section_config.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
            else:
                setattr(config, section_name, section_config)
                
        return config
    
    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        import yaml
        
        config_dict = self.to_dict()
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")
    
    @classmethod
    def load(cls, filepath: str) -> 'EngineGraphLoRAConfig':
        """Load configuration from file"""
        import json
        import yaml
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")
            
        return cls.from_dict(config_dict)


# Predefined configurations for common scenarios
def get_small_config() -> EngineGraphLoRAConfig:
    """Configuration for small-scale experiments"""
    config = EngineGraphLoRAConfig()
    
    # Smaller LLM
    config.llm.model_name = "distilbert-base-uncased"
    config.llm.hidden_size = 768
    config.llm.num_layers = 12
    
    # Reduced G-Ladder
    config.g_ladder.num_layers = 12
    config.g_ladder.hidden_size = 768
    config.g_ladder.lora_rank = 8
    
    # Smaller GNN
    config.gnn.num_layers = 2
    config.gnn.hidden_size = 128
    
    # Reduced training
    config.training.batch_size = 16
    config.training.learning_rate = 2e-4
    
    return config


def get_large_config() -> EngineGraphLoRAConfig:
    """Configuration for large-scale experiments"""
    config = EngineGraphLoRAConfig()
    
    # Larger LLM
    config.llm.model_name = "microsoft/DialoGPT-large"
    config.llm.hidden_size = 1280
    config.llm.num_layers = 36
    
    # Enhanced G-Ladder
    config.g_ladder.num_layers = 36
    config.g_ladder.hidden_size = 1280
    config.g_ladder.lora_rank = 32
    
    # Larger GNN
    config.gnn.num_layers = 4
    config.gnn.hidden_size = 512
    
    # Enhanced training
    config.training.batch_size = 64
    config.training.learning_rate = 5e-5
    
    return config


def get_debug_config() -> EngineGraphLoRAConfig:
    """Configuration for debugging and fast iteration"""
    config = get_small_config()
    
    # Debug settings
    config.debug = True
    config.log_level = "DEBUG"
    
    # Fast training
    config.training.num_epochs = 5
    config.training.batch_size = 4
    config.data.few_shot_episodes = 10
    
    # Reduced caching
    config.llm.cache_embeddings = False
    config.anchor.anchor_ratio = 0.1
    
    return config 