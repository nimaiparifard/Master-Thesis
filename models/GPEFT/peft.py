"""
Parameter Efficient Fine-Tuning (PEFT) Configuration Module

This module provides comprehensive PEFT configuration utilities for efficient fine-tuning
of large language models. It implements various PEFT methods including LoRA, Prefix Tuning,
Prompt Tuning, and IA3, with optimized configurations for different model architectures.

Key Features:
- Multiple PEFT methods (LoRA, Prefix, Prompt, IA3)
- Automatic model preparation for efficient training
- Optimized hyperparameter configurations
- Memory-efficient training setup
- Comprehensive error handling and validation

Performance Optimizations:
- INT8 quantization support for memory efficiency
- Optimized target module selection
- Efficient adapter initialization
- Memory-aware configuration parameters

Author: Graph Neural Network Research Team
Version: 2.0 (Optimized and Documented)
"""

from peft import (
    LoraConfig, 
    PrefixTuningConfig,
    PromptTuningConfig, 
    get_peft_model, 
    prepare_model_for_int8_training,
    TaskType, 
    IA3Config
)
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)


def create_peft_config(
    model: nn.Module,
    method: str = 'lora',
    task_type: TaskType = TaskType.SEQ_CLS,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: int = 32,
    dropout: float = 0.05,
    num_virtual_tokens: int = 20,
    use_int8: bool = True,
    **kwargs
) -> Tuple[nn.Module, Any]:
    """
    Create and apply PEFT configuration to a model for efficient fine-tuning.
    
    This function implements various Parameter Efficient Fine-Tuning (PEFT) methods
    that significantly reduce the number of trainable parameters while maintaining
    model performance. PEFT is crucial for fine-tuning large models on limited
    hardware resources.
    
    Supported PEFT Methods:
    - LoRA (Low-Rank Adaptation): Adds trainable low-rank matrices to attention layers
    - Prefix Tuning: Prepends trainable vectors to key and value matrices
    - Prompt Tuning: Adds trainable prompt tokens to the input sequence
    - IA3 (Infused Adapter by Inhibiting and Amplifying): Scales activations with learned vectors
    
    Algorithm Overview:
    1. Validate input parameters and model compatibility
    2. Create appropriate PEFT configuration based on method
    3. Prepare model for efficient training (optional INT8 quantization)
    4. Apply PEFT adapters to the model
    5. Display trainable parameter statistics
    
    Performance Benefits:
    - Reduces trainable parameters by 90-99%
    - Enables fine-tuning on consumer GPUs
    - Faster training and inference
    - Lower memory requirements with INT8 support
    
    Args:
        model (nn.Module): Base model to apply PEFT to
        method (str): PEFT method ('lora', 'prefix', 'prompt', 'ia3'). Default: 'lora'
        task_type (TaskType): Type of task for PEFT configuration. Default: SEQ_CLS
        target_modules (Optional[List[str]]): Specific modules to target. If None, uses defaults
        rank (int): Rank for LoRA decomposition. Default: 8
        alpha (int): LoRA scaling parameter. Default: 32
        dropout (float): Dropout rate for PEFT modules. Default: 0.05
        num_virtual_tokens (int): Number of virtual tokens for prefix/prompt tuning. Default: 20
        use_int8 (bool): Whether to use INT8 quantization. Default: True
        **kwargs: Additional method-specific parameters
    
    Returns:
        Tuple[nn.Module, Any]: PEFT-enabled model and PEFT configuration object
    
    Raises:
        ValueError: If method is not supported or parameters are invalid
        RuntimeError: If model preparation fails
    
    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained('bert-base-uncased')
        >>> peft_model, config = create_peft_config(model, method='lora', rank=16)
        >>> # Model now has ~0.1% trainable parameters instead of 100%
    
    Time Complexity: O(M) where M is number of model parameters
    Space Complexity: O(R * D) where R is rank, D is model dimension
    """
    # Input validation
    supported_methods = ['lora', 'prefix', 'prompt', 'ia3']
    if method.lower() not in supported_methods:
        raise ValueError(f"Method '{method}' not supported. Use one of: {supported_methods}")
    
    if rank <= 0:
        raise ValueError(f"Rank must be positive, got {rank}")
    
    if not 0 <= dropout <= 1:
        raise ValueError(f"Dropout must be in [0, 1], got {dropout}")
    
    logger.info(f"Creating PEFT configuration with method: {method}")
    logger.info(f"Parameters - rank: {rank}, alpha: {alpha}, dropout: {dropout}")
    
    # Get default target modules if not specified
    if target_modules is None:
        target_modules = _get_default_target_modules(model, method)
    
    # Create PEFT configuration based on method
    method = method.lower()
    
    if method == 'lora':
        peft_config = _create_lora_config(
            task_type=task_type,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            **kwargs
        )
    
    elif method == 'prefix':
        peft_config = _create_prefix_config(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
            **kwargs
        )
    
    elif method == 'prompt':
        peft_config = _create_prompt_config(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
            **kwargs
        )
    
    elif method == 'ia3':
        peft_config = _create_ia3_config(
            task_type=task_type,
            target_modules=target_modules,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unexpected method: {method}")
    
    try:
        # Prepare model for efficient training
        if use_int8:
            logger.info("Preparing model for INT8 training")
            model = prepare_model_for_int8_training(model)
        
        # Apply PEFT configuration
        logger.info("Applying PEFT adapters to model")
        peft_model = get_peft_model(model, peft_config)
        
        # Display parameter statistics
        _log_parameter_statistics(peft_model)
        
        return peft_model, peft_config
    
    except Exception as e:
        logger.error(f"Failed to create PEFT model: {e}")
        raise RuntimeError(f"PEFT model creation failed: {e}")


def _create_lora_config(
    task_type: TaskType,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
    **kwargs
) -> LoraConfig:
    """
    Create LoRA (Low-Rank Adaptation) configuration.
    
    LoRA works by adding trainable low-rank matrices to existing linear layers,
    significantly reducing the number of trainable parameters while maintaining
    model expressiveness.
    
    Args:
        task_type (TaskType): Type of task
        rank (int): Rank of adaptation matrices
        alpha (int): Scaling parameter
        dropout (float): Dropout rate
        target_modules (List[str]): Modules to apply LoRA to
        **kwargs: Additional LoRA parameters
    
    Returns:
        LoraConfig: Configured LoRA configuration object
    """
    # LoRA-specific parameter validation
    bias = kwargs.get('bias', 'none')
    fan_in_fan_out = kwargs.get('fan_in_fan_out', False)
    
    return LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        fan_in_fan_out=fan_in_fan_out
    )


def _create_prefix_config(
    task_type: TaskType,
    num_virtual_tokens: int,
    **kwargs
) -> PrefixTuningConfig:
    """
    Create Prefix Tuning configuration.
    
    Prefix Tuning prepends trainable vectors to the key and value matrices
    in attention layers, allowing task-specific adaptation without modifying
    the base model parameters.
    
    Args:
        task_type (TaskType): Type of task
        num_virtual_tokens (int): Number of virtual tokens to prepend
        **kwargs: Additional prefix tuning parameters
    
    Returns:
        PrefixTuningConfig: Configured prefix tuning configuration
    """
    encoder_hidden_size = kwargs.get('encoder_hidden_size', None)
    prefix_projection = kwargs.get('prefix_projection', False)
    
    config_args = {
        'task_type': task_type,
        'inference_mode': False,
        'num_virtual_tokens': num_virtual_tokens
    }
    
    if encoder_hidden_size is not None:
        config_args['encoder_hidden_size'] = encoder_hidden_size
    
    if prefix_projection:
        config_args['prefix_projection'] = prefix_projection
    
    return PrefixTuningConfig(**config_args)


def _create_prompt_config(
    task_type: TaskType,
    num_virtual_tokens: int,
    **kwargs
) -> PromptTuningConfig:
    """
    Create Prompt Tuning configuration.
    
    Prompt Tuning adds trainable prompt tokens to the input sequence,
    allowing task adaptation through learned prompt representations.
    
    Args:
        task_type (TaskType): Type of task
        num_virtual_tokens (int): Number of prompt tokens
        **kwargs: Additional prompt tuning parameters
    
    Returns:
        PromptTuningConfig: Configured prompt tuning configuration
    """
    prompt_tuning_init = kwargs.get('prompt_tuning_init', 'RANDOM')
    prompt_tuning_init_text = kwargs.get('prompt_tuning_init_text', None)
    tokenizer_name_or_path = kwargs.get('tokenizer_name_or_path', None)
    
    config_args = {
        'task_type': task_type,
        'inference_mode': False,
        'num_virtual_tokens': num_virtual_tokens,
        'prompt_tuning_init': prompt_tuning_init
    }
    
    if prompt_tuning_init_text is not None:
        config_args['prompt_tuning_init_text'] = prompt_tuning_init_text
    
    if tokenizer_name_or_path is not None:
        config_args['tokenizer_name_or_path'] = tokenizer_name_or_path
    
    return PromptTuningConfig(**config_args)


def _create_ia3_config(
    task_type: TaskType,
    target_modules: List[str],
    **kwargs
) -> IA3Config:
    """
    Create IA3 (Infused Adapter by Inhibiting and Amplifying) configuration.
    
    IA3 introduces learned scaling vectors that are element-wise multiplied
    with activations, providing a parameter-efficient way to adapt models.
    
    Args:
        task_type (TaskType): Type of task
        target_modules (List[str]): Modules to apply IA3 to
        **kwargs: Additional IA3 parameters
    
    Returns:
        IA3Config: Configured IA3 configuration
    """
    feedforward_modules = kwargs.get('feedforward_modules', None)
    
    config_args = {
        'task_type': task_type,
        'inference_mode': False,
        'target_modules': target_modules
    }
    
    if feedforward_modules is not None:
        config_args['feedforward_modules'] = feedforward_modules
    
    return IA3Config(**config_args)


def _get_default_target_modules(model: nn.Module, method: str) -> List[str]:
    """
    Get default target modules based on model architecture and PEFT method.
    
    Args:
        model (nn.Module): Model to analyze
        method (str): PEFT method
    
    Returns:
        List[str]: Default target module names
    """
    # Common target modules for different architectures
    common_targets = {
        'bert': ['query', 'value', 'key', 'dense'],
        'roberta': ['query', 'value', 'key', 'dense'],
        'gpt2': ['c_attn', 'c_proj'],
        'llama': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'mistral': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'default': ['q_proj', 'v_proj']  # Conservative default
    }
    
    # Detect model architecture
    model_class_name = model.__class__.__name__.lower()
    
    for arch_name, modules in common_targets.items():
        if arch_name in model_class_name:
            logger.info(f"Detected {arch_name} architecture, using target modules: {modules}")
            return modules
    
    # Method-specific defaults
    if method in ['lora', 'ia3']:
        default_modules = common_targets['default']
    else:
        default_modules = []  # Prefix and prompt tuning don't need target modules
    
    logger.info(f"Using default target modules for {method}: {default_modules}")
    return default_modules


def _log_parameter_statistics(model: nn.Module) -> None:
    """
    Log detailed parameter statistics for the PEFT model.
    
    Args:
        model (nn.Module): PEFT model to analyze
    """
    if hasattr(model, 'print_trainable_parameters'):
        # Use PEFT's built-in method
        model.print_trainable_parameters()
    else:
        # Manual calculation
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage of trainable parameters: {percentage:.2f}%")


def optimize_peft_config_for_hardware(
    base_config: Dict[str, Any],
    available_memory_gb: float,
    model_size_gb: float
) -> Dict[str, Any]:
    """
    Optimize PEFT configuration based on available hardware resources.
    
    This function adjusts PEFT parameters to fit within memory constraints
    while maintaining optimal performance characteristics.
    
    Args:
        base_config (Dict[str, Any]): Base PEFT configuration
        available_memory_gb (float): Available GPU memory in GB
        model_size_gb (float): Base model size in GB
    
    Returns:
        Dict[str, Any]: Optimized configuration
    """
    optimized_config = base_config.copy()
    
    # Memory-based optimization heuristics
    memory_ratio = available_memory_gb / model_size_gb
    
    if memory_ratio < 2.0:
        # Very limited memory - use most aggressive settings
        optimized_config.update({
            'rank': min(optimized_config.get('rank', 8), 4),
            'use_int8': True,
            'dropout': max(optimized_config.get('dropout', 0.05), 0.1)
        })
        logger.info("Applied aggressive memory optimization")
    
    elif memory_ratio < 4.0:
        # Moderate memory - balanced settings
        optimized_config.update({
            'rank': min(optimized_config.get('rank', 8), 8),
            'use_int8': True
        })
        logger.info("Applied moderate memory optimization")
    
    else:
        # Abundant memory - performance-focused settings
        optimized_config.update({
            'rank': min(optimized_config.get('rank', 8), 16),
            'use_int8': False
        })
        logger.info("Applied performance-focused optimization")
    
    return optimized_config


def validate_peft_compatibility(model: nn.Module, method: str) -> bool:
    """
    Validate if a model is compatible with the specified PEFT method.
    
    Args:
        model (nn.Module): Model to validate
        method (str): PEFT method to check compatibility for
    
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        # Check for required attributes/methods based on PEFT method
        if method.lower() in ['lora', 'ia3']:
            # These methods need linear layers
            has_linear = any(isinstance(module, nn.Linear) for module in model.modules())
            if not has_linear:
                logger.warning(f"Model has no Linear layers, {method} may not be effective")
                return False
        
        # Check for embedding layers (needed for prompt/prefix tuning)
        if method.lower() in ['prompt', 'prefix']:
            has_embeddings = any('embed' in name.lower() for name, _ in model.named_modules())
            if not has_embeddings:
                logger.warning(f"Model may not have embedding layers, {method} compatibility uncertain")
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating PEFT compatibility: {e}")
        return False