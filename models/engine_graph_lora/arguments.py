"""
ENGINE-GraphLoRA Arguments Configuration

This module provides comprehensive argument parsing and configuration management
for the ENGINE-GraphLoRA framework. It combines all necessary parameters for
training, evaluation, and inference across all components.

Usage:
    python main.py --config config.yaml
    python main.py --model_name bert-base-uncased --num_epochs 100
    
Arguments are organized into logical groups:
- Model Configuration
- Training Parameters  
- Data Configuration
- Optimization Settings
- Evaluation Settings
- System Configuration

Author: Graph Neural Network Research Team
Version: 1.0
"""

import argparse
import yaml
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch

from .config import EngineGraphLoRAConfig


class EngineGraphLoRAArguments:
    """
    Comprehensive argument parser for ENGINE-GraphLoRA framework
    
    Supports both command-line arguments and configuration files.
    Arguments can be loaded from YAML/JSON files or passed via command line.
    Command line arguments override configuration file values.
    """
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="ENGINE-GraphLoRA: Text-Graph Learning with Parameter-Efficient Adaptation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._setup_arguments()
    
    def _setup_arguments(self):
        """Setup all argument groups and parameters"""
        self._add_general_args()
        self._add_model_args()
        self._add_data_args()
        self._add_training_args()
        self._add_evaluation_args()
        self._add_optimization_args()
        self._add_system_args()
        self._add_experiment_args()
    
    def _add_general_args(self):
        """General configuration arguments"""
        general = self.parser.add_argument_group('General Configuration')
        
        general.add_argument(
            '--config', type=str, default=None,
            help='Path to configuration file (YAML/JSON)'
        )
        general.add_argument(
            '--task', type=str, default='node_classification',
            choices=['node_classification', 'link_prediction', 'graph_classification'],
            help='Task type for the model'
        )
        general.add_argument(
            '--dataset', type=str, required=True,
            help='Dataset name (e.g., cora, citeseer, arxiv)'
        )
        general.add_argument(
            '--output_dir', type=str, default='./outputs',
            help='Output directory for results and checkpoints'
        )
        general.add_argument(
            '--seed', type=int, default=42,
            help='Random seed for reproducibility'
        )
        general.add_argument(
            '--debug', action='store_true',
            help='Enable debug mode'
        )
    
    def _add_model_args(self):
        """Model architecture arguments"""
        model = self.parser.add_argument_group('Model Configuration')
        
        # LLM Configuration
        model.add_argument(
            '--llm_model_name', type=str, default='bert-base-uncased',
            help='Pre-trained language model name'
        )
        model.add_argument(
            '--llm_hidden_size', type=int, default=768,
            help='Hidden size of the language model'
        )
        model.add_argument(
            '--llm_num_layers', type=int, default=12,
            help='Number of layers in the language model'
        )
        model.add_argument(
            '--llm_freeze_base', action='store_true',
            help='Freeze base LLM parameters'
        )
        model.add_argument(
            '--llm_gradient_checkpointing', action='store_true',
            help='Enable gradient checkpointing for memory efficiency'
        )
        
        # G-Ladder Configuration
        model.add_argument(
            '--g_ladder_num_layers', type=int, default=12,
            help='Number of G-Ladder layers'
        )
        model.add_argument(
            '--g_ladder_hidden_size', type=int, default=768,
            help='Hidden size for G-Ladder layers'
        )
        model.add_argument(
            '--g_ladder_injection_layers', type=int, nargs='+', default=[3, 6, 9],
            help='Layers where G-Ladder injects graph information'
        )
        model.add_argument(
            '--g_ladder_cross_attention', action='store_true',
            help='Use cross-attention in G-Ladder'
        )
        model.add_argument(
            '--g_ladder_residual_connections', action='store_true',
            help='Use residual connections in G-Ladder'
        )
        model.add_argument(
            '--g_ladder_layer_norm', action='store_true',
            help='Apply layer normalization in G-Ladder'
        )
        
        # LoRA Configuration
        model.add_argument(
            '--lora_rank', type=int, default=16,
            help='Rank for LoRA decomposition'
        )
        model.add_argument(
            '--lora_alpha', type=float, default=32.0,
            help='Scaling factor for LoRA'
        )
        model.add_argument(
            '--lora_dropout', type=float, default=0.1,
            help='Dropout rate for LoRA layers'
        )
        model.add_argument(
            '--lora_target_modules', type=str, nargs='+', 
            default=['query', 'key', 'value', 'dense'],
            help='Target modules for LoRA adaptation'
        )
        
        # GNN Configuration
        model.add_argument(
            '--gnn_type', type=str, default='sage',
            choices=['sage', 'gat', 'gcn', 'gin'],
            help='Type of GNN to use'
        )
        model.add_argument(
            '--gnn_hidden_size', type=int, default=256,
            help='Hidden size for GNN layers'
        )
        model.add_argument(
            '--gnn_num_layers', type=int, default=3,
            help='Number of GNN layers'
        )
        model.add_argument(
            '--gnn_dropout', type=float, default=0.2,
            help='Dropout rate for GNN layers'
        )
        model.add_argument(
            '--gnn_activation', type=str, default='relu',
            choices=['relu', 'gelu', 'swish', 'tanh'],
            help='Activation function for GNN'
        )
        
        # Anchor System Configuration
        model.add_argument(
            '--anchor_selection_strategy', type=str, default='centrality',
            choices=['centrality', 'degree', 'random', 'uncertainty'],
            help='Strategy for selecting anchor nodes'
        )
        model.add_argument(
            '--anchor_ratio', type=float, default=0.1,
            help='Ratio of nodes to use as anchors'
        )
        model.add_argument(
            '--anchor_budget_ratio', type=float, default=0.3,
            help='Budget ratio for LLM refresh on anchors'
        )
        model.add_argument(
            '--cache_size', type=int, default=10000,
            help='Size of the embedding cache'
        )
        model.add_argument(
            '--cache_update_frequency', type=int, default=5,
            help='Frequency of cache updates (in epochs)'
        )
    
    def _add_data_args(self):
        """Data configuration arguments"""
        data = self.parser.add_argument_group('Data Configuration')
        
        data.add_argument(
            '--data_dir', type=str, default='./data',
            help='Directory containing the dataset'
        )
        data.add_argument(
            '--max_text_length', type=int, default=512,
            help='Maximum text sequence length'
        )
        data.add_argument(
            '--text_field', type=str, default='text',
            help='Field name containing text data'
        )
        data.add_argument(
            '--train_ratio', type=float, default=0.6,
            help='Training data ratio'
        )
        data.add_argument(
            '--val_ratio', type=float, default=0.2,
            help='Validation data ratio'
        )
        data.add_argument(
            '--test_ratio', type=float, default=0.2,
            help='Test data ratio'
        )
        data.add_argument(
            '--num_workers', type=int, default=4,
            help='Number of data loading workers'
        )
        data.add_argument(
            '--prefetch_factor', type=int, default=2,
            help='Prefetch factor for data loading'
        )
    
    def _add_training_args(self):
        """Training configuration arguments"""
        training = self.parser.add_argument_group('Training Configuration')
        
        training.add_argument(
            '--num_epochs', type=int, default=100,
            help='Number of training epochs'
        )
        training.add_argument(
            '--batch_size', type=int, default=32,
            help='Training batch size'
        )
        training.add_argument(
            '--learning_rate', type=float, default=1e-4,
            help='Learning rate'
        )
        training.add_argument(
            '--weight_decay', type=float, default=1e-5,
            help='Weight decay for regularization'
        )
        training.add_argument(
            '--optimizer', type=str, default='adamw',
            choices=['adamw', 'adam', 'sgd'],
            help='Optimizer type'
        )
        training.add_argument(
            '--scheduler', type=str, default='cosine',
            choices=['cosine', 'linear', 'constant', 'polynomial'],
            help='Learning rate scheduler'
        )
        training.add_argument(
            '--warmup_steps', type=int, default=1000,
            help='Number of warmup steps'
        )
        training.add_argument(
            '--gradient_clip_norm', type=float, default=1.0,
            help='Gradient clipping norm'
        )
        training.add_argument(
            '--early_stopping_patience', type=int, default=20,
            help='Early stopping patience'
        )
        training.add_argument(
            '--validation_frequency', type=int, default=1,
            help='Validation frequency (in epochs)'
        )
        training.add_argument(
            '--checkpoint_frequency', type=int, default=10,
            help='Checkpoint saving frequency (in epochs)'
        )
        training.add_argument(
            '--use_fp16', action='store_true',
            help='Use mixed precision training'
        )
    
    def _add_evaluation_args(self):
        """Evaluation configuration arguments"""
        eval_group = self.parser.add_argument_group('Evaluation Configuration')
        
        eval_group.add_argument(
            '--eval_batch_size', type=int, default=64,
            help='Batch size for evaluation'
        )
        eval_group.add_argument(
            '--eval_metrics', type=str, nargs='+',
            default=['accuracy', 'f1', 'precision', 'recall'],
            help='Evaluation metrics to compute'
        )
        eval_group.add_argument(
            '--eval_early_exit', action='store_true',
            help='Enable early exit during evaluation'
        )
        eval_group.add_argument(
            '--eval_confidence_threshold', type=float, default=0.8,
            help='Confidence threshold for early exit'
        )
        eval_group.add_argument(
            '--eval_detailed_metrics', action='store_true',
            help='Compute detailed evaluation metrics'
        )
        eval_group.add_argument(
            '--eval_save_predictions', action='store_true',
            help='Save predictions during evaluation'
        )
    
    def _add_optimization_args(self):
        """Optimization and loss configuration arguments"""
        opt = self.parser.add_argument_group('Optimization Configuration')
        
        # Loss weights
        opt.add_argument(
            '--task_loss_weight', type=float, default=1.0,
            help='Weight for task-specific loss'
        )
        opt.add_argument(
            '--smmd_loss_weight', type=float, default=0.1,
            help='Weight for SMMD loss (λ in the algorithm)'
        )
        opt.add_argument(
            '--contrastive_loss_weight', type=float, default=0.05,
            help='Weight for contrastive loss (μ in the algorithm)'
        )
        
        # SMMD Loss Configuration
        opt.add_argument(
            '--smmd_kernel', type=str, default='rbf',
            choices=['rbf', 'linear', 'polynomial'],
            help='Kernel type for SMMD loss'
        )
        opt.add_argument(
            '--smmd_bandwidth', type=float, default=1.0,
            help='Bandwidth for SMMD kernel'
        )
        opt.add_argument(
            '--smmd_include_degree', action='store_true',
            help='Include degree information in SMMD'
        )
        opt.add_argument(
            '--smmd_include_clustering', action='store_true',
            help='Include clustering coefficient in SMMD'
        )
        
        # Contrastive Loss Configuration
        opt.add_argument(
            '--contrastive_temperature', type=float, default=0.07,
            help='Temperature for contrastive loss'
        )
        opt.add_argument(
            '--negative_sampling_ratio', type=float, default=1.0,
            help='Negative sampling ratio for contrastive loss'
        )
        opt.add_argument(
            '--hard_negative_mining', action='store_true',
            help='Enable hard negative mining'
        )
    
    def _add_system_args(self):
        """System configuration arguments"""
        system = self.parser.add_argument_group('System Configuration')
        
        system.add_argument(
            '--device', type=str, default='auto',
            choices=['auto', 'cpu', 'cuda', 'mps'],
            help='Device to use for training'
        )
        system.add_argument(
            '--gpu_ids', type=int, nargs='+', default=None,
            help='GPU IDs to use for training'
        )
        system.add_argument(
            '--distributed', action='store_true',
            help='Enable distributed training'
        )
        system.add_argument(
            '--log_level', type=str, default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            help='Logging level'
        )
        system.add_argument(
            '--log_file', type=str, default=None,
            help='Log file path'
        )
    
    def _add_experiment_args(self):
        """Experiment tracking arguments"""
        exp = self.parser.add_argument_group('Experiment Tracking')
        
        exp.add_argument(
            '--experiment_name', type=str, default='engine_graph_lora',
            help='Experiment name'
        )
        exp.add_argument(
            '--run_name', type=str, default=None,
            help='Run name for this experiment'
        )
        exp.add_argument(
            '--wandb_project', type=str, default=None,
            help='Weights & Biases project name'
        )
        exp.add_argument(
            '--wandb_entity', type=str, default=None,
            help='Weights & Biases entity name'
        )
        exp.add_argument(
            '--wandb_tags', type=str, nargs='+', default=None,
            help='Tags for the wandb run'
        )
        exp.add_argument(
            '--save_model', action='store_true',
            help='Save the trained model'
        )
        exp.add_argument(
            '--save_predictions', action='store_true',
            help='Save model predictions'
        )
    
    def parse_args(self, args=None) -> EngineGraphLoRAConfig:
        """
        Parse arguments and return configuration object
        
        Args:
            args: Optional list of arguments (for testing)
            
        Returns:
            EngineGraphLoRAConfig object with parsed arguments
        """
        parsed_args = self.parser.parse_args(args)
        
        # Load configuration from file if provided
        config_dict = {}
        if parsed_args.config:
            config_dict = self._load_config_file(parsed_args.config)
        
        # Override with command line arguments
        cli_dict = self._args_to_dict(parsed_args)
        config_dict.update(cli_dict)
        
        # Convert to configuration object
        config = self._dict_to_config(config_dict)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def _args_to_dict(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert parsed arguments to nested dictionary"""
        args_dict = vars(args)
        
        # Remove None values and config file path
        args_dict = {k: v for k, v in args_dict.items() if v is not None and k != 'config'}
        
        # Group arguments by component
        grouped_dict = {
            'llm': {},
            'g_ladder': {},
            'lora': {},
            'gnn': {},
            'anchor': {},
            'loss': {},
            'training': {},
            'data': {},
            'inference': {},
            'cross_modal': {}
        }
        
        # Map arguments to appropriate groups
        for key, value in args_dict.items():
            if key.startswith('llm_'):
                grouped_dict['llm'][key[4:]] = value
            elif key.startswith('g_ladder_'):
                # Map g_ladder arguments with proper names
                g_key = key[9:]  # Remove 'g_ladder_' prefix
                if g_key == 'cross_attention':
                    # Skip this parameter as it's not in GLadderConfig
                    continue
                elif g_key == 'residual_connections':
                    # Skip this parameter as it's not in GLadderConfig
                    continue
                elif g_key == 'layer_norm':
                    # Skip this parameter as it's not in GLadderConfig
                    continue
                else:
                    grouped_dict['g_ladder'][g_key] = value
            elif key.startswith('lora_'):
                grouped_dict['lora'][key[5:]] = value
            elif key.startswith('gnn_'):
                grouped_dict['gnn'][key[4:]] = value
            elif key.startswith('anchor_') or key.startswith('cache_'):
                grouped_dict['anchor'][key] = value
            elif key.startswith('smmd_') or key.startswith('contrastive_') or 'loss_weight' in key:
                grouped_dict['loss'][key] = value
            elif key in ['num_epochs', 'batch_size', 'learning_rate', 'weight_decay', 
                        'optimizer', 'scheduler', 'warmup_steps', 'gradient_clip_norm',
                        'early_stopping_patience', 'validation_frequency', 'checkpoint_frequency',
                        'use_fp16']:
                grouped_dict['training'][key] = value
            elif key.startswith('data_') or key in ['max_text_length', 'text_field', 
                                                   'train_ratio', 'val_ratio', 'test_ratio',
                                                   'num_workers', 'prefetch_factor']:
                grouped_dict['data'][key] = value
            elif key.startswith('eval_'):
                grouped_dict['inference'][key[5:]] = value
            else:
                # Global arguments
                grouped_dict[key] = value
        
        return grouped_dict
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> EngineGraphLoRAConfig:
        """Convert configuration dictionary to EngineGraphLoRAConfig object"""
        
        # Handle device setting
        if config_dict.get('device') == 'auto':
            if torch.cuda.is_available():
                config_dict['device'] = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                config_dict['device'] = 'mps'
            else:
                config_dict['device'] = 'cpu'
        
        # Create config object with proper nesting
        try:
            # Import config classes
            from .config import (
                LLMConfig, GLadderConfig, LoRAConfig, GNNConfig, AnchorConfig,
                LossConfig, CrossModalConfig, TrainingConfig, DataConfig, InferenceConfig
            )
            
            # Extract global keys and nested component keys
            global_keys = {}
            component_configs = {}
            
            for key, value in config_dict.items():
                if key == 'llm' and isinstance(value, dict):
                    component_configs['llm'] = LLMConfig(**value)
                elif key == 'g_ladder' and isinstance(value, dict):
                    component_configs['g_ladder'] = GLadderConfig(**value)
                elif key == 'lora' and isinstance(value, dict):
                    component_configs['lora'] = LoRAConfig(**value)
                elif key == 'gnn' and isinstance(value, dict):
                    component_configs['gnn'] = GNNConfig(**value)
                elif key == 'anchor' and isinstance(value, dict):
                    component_configs['anchor'] = AnchorConfig(**value)
                elif key == 'loss' and isinstance(value, dict):
                    component_configs['loss'] = LossConfig(**value)
                elif key == 'cross_modal' and isinstance(value, dict):
                    component_configs['cross_modal'] = CrossModalConfig(**value)
                elif key == 'training' and isinstance(value, dict):
                    component_configs['training'] = TrainingConfig(**value)
                elif key == 'data' and isinstance(value, dict):
                    component_configs['data'] = DataConfig(**value)
                elif key == 'inference' and isinstance(value, dict):
                    component_configs['inference'] = InferenceConfig(**value)
                elif key in ['llm', 'g_ladder', 'lora', 'gnn', 'anchor', 'loss', 
                           'training', 'data', 'inference', 'cross_modal']:
                    # Keep as is if not dict
                    component_configs[key] = value
                else:
                    global_keys[key] = value
            
            # Create config with components
            config = EngineGraphLoRAConfig(**component_configs)
            
            # Set global attributes
            for key, value in global_keys.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
        except Exception as e:
            print(f"Error creating config: {e}")
            print(f"Config dict keys: {list(config_dict.keys())}")
            # Fallback to default config with updates
            config = EngineGraphLoRAConfig()
            
            # Import config classes for fallback
            from .config import (
                LLMConfig, GLadderConfig, LoRAConfig, GNNConfig, AnchorConfig,
                LossConfig, CrossModalConfig, TrainingConfig, DataConfig, InferenceConfig
            )
            
            # Try to set attributes carefully
            for key, value in config_dict.items():
                try:
                    if hasattr(config, key):
                        setattr(config, key, value)
                    elif key in ['llm', 'g_ladder', 'lora', 'gnn', 'anchor', 'loss', 
                               'training', 'data', 'inference', 'cross_modal']:
                        # Try to update nested config
                        nested_config = getattr(config, key)
                        
                        if isinstance(value, dict) and hasattr(nested_config, '__dict__'):
                            # Create new config object with valid parameters only
                            import dataclasses
                            
                            config_classes = {
                                'llm': LLMConfig,
                                'g_ladder': GLadderConfig,
                                'lora': LoRAConfig,
                                'gnn': GNNConfig,
                                'anchor': AnchorConfig,
                                'loss': LossConfig,
                                'training': TrainingConfig,
                                'data': DataConfig,
                                'inference': InferenceConfig,
                                'cross_modal': CrossModalConfig
                            }
                            
                            if key in config_classes:
                                config_class = config_classes[key]
                                # Get valid field names from dataclass
                                field_names = {f.name for f in dataclasses.fields(config_class)}
                                valid_params = {k: v for k, v in value.items() if k in field_names}
                                
                                if valid_params:
                                    try:
                                        setattr(config, key, config_class(**valid_params))
                                    except Exception as create_e:
                                        print(f"Could not create {key} config: {create_e}")
                                        # Fall back to updating individual attributes
                                        for nested_key, nested_value in valid_params.items():
                                            if hasattr(nested_config, nested_key):
                                                setattr(nested_config, nested_key, nested_value)
                        else:
                            # Fallback: update existing nested config attributes
                            for nested_key, nested_value in value.items():
                                if hasattr(nested_config, nested_key):
                                    setattr(nested_config, nested_key, nested_value)
                except Exception as nested_e:
                    print(f"Warning: Could not set {key}={value}: {nested_e}")
        
        return config
    
    def _validate_config(self, config: EngineGraphLoRAConfig):
        """Validate configuration parameters"""
        
        try:
            # Check ratios sum to 1
            if hasattr(config, 'data') and hasattr(config.data, 'train_ratio'):
                total_ratio = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
                if abs(total_ratio - 1.0) > 1e-6:
                    raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")
            
            # Check positive values
            if hasattr(config, 'training') and hasattr(config.training, 'learning_rate'):
                if config.training.learning_rate <= 0:
                    raise ValueError("Learning rate must be positive")
            
            if hasattr(config, 'training') and hasattr(config.training, 'batch_size'):
                if config.training.batch_size <= 0:
                    raise ValueError("Batch size must be positive")
            
            if hasattr(config, 'lora') and hasattr(config.lora, 'rank'):
                if config.lora.rank <= 0:
                    raise ValueError("LoRA rank must be positive")
            
            if hasattr(config, 'anchor') and hasattr(config.anchor, 'anchor_ratio'):
                if config.anchor.anchor_ratio <= 0 or config.anchor.anchor_ratio > 1:
                    raise ValueError("Anchor ratio must be between 0 and 1")
            
            # Check device availability
            if hasattr(config, 'device'):
                if config.device == 'cuda' and not torch.cuda.is_available():
                    print("Warning: CUDA requested but not available, falling back to CPU")
                    config.device = 'cpu'
            
            print(f"Configuration validated successfully")
            print(f"Device: {getattr(config, 'device', 'cpu')}")
            print(f"Model: {getattr(config.llm, 'model_name', 'Not specified') if hasattr(config, 'llm') else 'Not specified'}")
            print(f"Dataset: {getattr(config, 'dataset', 'Not specified')}")
            
        except Exception as e:
            print(f"Warning: Configuration validation failed: {e}")
            # Don't raise error to allow fallback behavior


def get_default_config() -> EngineGraphLoRAConfig:
    """Get default configuration"""
    return EngineGraphLoRAConfig()


def save_config(config: EngineGraphLoRAConfig, path: str):
    """Save configuration to file"""
    config_dict = config.to_dict()
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix.lower() in ['.yaml', '.yml']:
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif path.suffix.lower() == '.json':
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_config(path: str) -> EngineGraphLoRAConfig:
    """Load configuration from file"""
    args = EngineGraphLoRAArguments()
    return args.parse_args(['--config', path])


if __name__ == "__main__":
    # Example usage
    args = EngineGraphLoRAArguments()
    config = args.parse_args()
    print("Configuration loaded successfully!")
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.device}") 