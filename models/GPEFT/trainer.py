"""
Advanced Language Model Training Module

This module provides comprehensive training utilities for language models with support
for BERT-based classifiers, evaluation metrics, and efficient training strategies.
It implements sophisticated training pipelines with memory optimization, advanced
evaluation metrics, and flexible configuration management.

Key Features:
- BERT-based classification with custom architectures
- Comprehensive evaluation metrics and monitoring
- Memory-efficient training with mixed precision support
- Flexible dataset handling and preprocessing
- Advanced training strategies with early stopping
- Model checkpointing and state management

Performance Optimizations:
- Mixed precision training for memory efficiency
- Gradient accumulation for large effective batch sizes
- Efficient data loading with parallel workers
- Memory-mapped dataset storage for large corpora
- Optimized evaluation with larger batch sizes

Author: Graph Neural Network Research Team
Version: 2.0 (Optimized and Documented)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer, 
    IntervalStrategy,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time
import logging
from pathlib import Path
import json
from collections import defaultdict
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    model_name: str = 'bert-base-uncased'
    feat_shrink: Optional[int] = None
    dropout: float = 0.1
    attention_dropout: float = 0.1
    classifier_dropout: float = 0.1
    num_labels: int = 2
    use_pooler: bool = False


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_epochs: float = 0.1
    eval_patience: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    save_strategy: str = 'steps'
    evaluation_strategy: str = 'steps'
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500


def compute_comprehensive_metrics(predictions_and_labels: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for classification tasks.
    
    This function calculates multiple evaluation metrics including accuracy,
    precision, recall, F1-score, and additional statistics to provide a
    complete picture of model performance.
    
    Metrics Computed:
    - Accuracy: Overall classification accuracy
    - Precision: Per-class and macro-averaged precision
    - Recall: Per-class and macro-averaged recall
    - F1-Score: Per-class and macro-averaged F1-score
    - Support: Number of samples per class
    
    Args:
        predictions_and_labels (Tuple): Tuple containing (predictions, labels)
            - predictions (np.ndarray): Model predictions of shape [num_samples, num_classes]
            - labels (np.ndarray): True labels of shape [num_samples]
    
    Returns:
        Dict[str, float]: Dictionary containing all computed metrics
    
    Example:
        >>> predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        >>> labels = np.array([1, 0, 1])
        >>> metrics = compute_comprehensive_metrics((predictions, labels))
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    Time Complexity: O(n * c) where n=samples, c=classes
    Space Complexity: O(c^2) for confusion matrix
    """
    predictions, labels = predictions_and_labels
    
    # Convert predictions to class indices
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Basic accuracy
    accuracy = accuracy_score(y_true=labels, y_pred=predicted_labels)
    
    # Precision, recall, F1-score with different averaging strategies
    precision_macro, recall_macro, f1_macro, support = precision_recall_fscore_support(
        labels, predicted_labels, average='macro', zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predicted_labels, average='micro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predicted_labels, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, predicted_labels, average=None, zero_division=0
    )
    
    # Compile comprehensive metrics
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }
    
    # Add per-class metrics
    for i, (prec, rec, f1, supp) in enumerate(zip(
        precision_per_class, recall_per_class, f1_per_class, support_per_class
    )):
        metrics.update({
            f'precision_class_{i}': prec,
            f'recall_class_{i}': rec,
            f'f1_class_{i}': f1,
            f'support_class_{i}': supp
        })
    
    return metrics


class BertClassifier(nn.Module):
    """
    Advanced BERT-based classifier with flexible architecture options.
    
    This class implements a sophisticated BERT-based classification model
    with support for feature dimensionality reduction, multiple dropout
    strategies, and flexible output configurations.
    """
    
    def __init__(
        self,
        bert_model: nn.Module,
        num_labels: int,
        feat_shrink: Optional[int] = None,
        dropout: float = 0.1,
        classifier_dropout: float = 0.1,
        use_pooler: bool = False,
        hidden_activation: str = 'gelu'
    ):
        """
        Initialize the BERT classifier.
        
        Args:
            bert_model (nn.Module): Pre-trained BERT model
            num_labels (int): Number of output classes
            feat_shrink (Optional[int]): Hidden dimension for feature reduction. If None, uses BERT's hidden size
            dropout (float): Dropout rate for BERT layers. Default: 0.1
            classifier_dropout (float): Dropout rate for classifier head. Default: 0.1
            use_pooler (bool): Whether to use BERT's pooler output. Default: False
            hidden_activation (str): Activation function for hidden layers. Default: 'gelu'
        """
        super(BertClassifier, self).__init__()
        
        self.bert = bert_model
        self.num_labels = num_labels
        self.use_pooler = use_pooler
        
        # Get BERT's hidden size
        self.bert_hidden_size = bert_model.config.hidden_size
        
        # Feature dimensionality reduction
        if feat_shrink is not None and feat_shrink != self.bert_hidden_size:
            self.feature_reducer = nn.Sequential(
                nn.Linear(self.bert_hidden_size, feat_shrink),
                self._get_activation(hidden_activation),
                nn.Dropout(dropout)
            )
            classifier_input_size = feat_shrink
        else:
            self.feature_reducer = None
            classifier_input_size = self.bert_hidden_size
        
        # Classification head
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"BertClassifier initialized: {self.bert_hidden_size} -> "
                   f"{classifier_input_size} -> {num_labels}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation_name.lower(), nn.GELU())
    
    def _init_weights(self):
        """Initialize classifier weights."""
        if self.feature_reducer is not None:
            for module in self.feature_reducer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the BERT classifier.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (Optional[torch.Tensor]): Attention mask
            token_type_ids (Optional[torch.Tensor]): Token type IDs
            return_features (bool): Whether to return intermediate features
        
        Returns:
            Union[torch.Tensor, Tuple]: Classification logits or (logits, features)
        """
        # BERT forward pass
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Extract features
        if self.use_pooler and hasattr(bert_outputs, 'pooler_output'):
            features = bert_outputs.pooler_output
        else:
            # Use [CLS] token representation
            features = bert_outputs.last_hidden_state[:, 0, :]
        
        # Apply feature reduction if configured
        if self.feature_reducer is not None:
            features = self.feature_reducer(features)
        
        # Classification
        features_dropped = self.classifier_dropout(features)
        logits = self.classifier(features_dropped)
        
        if return_features:
            return logits, features
        return logits


class BertClassificationInferenceModel(nn.Module):
    """
    Optimized BERT model for inference with memory-mapped output storage.
    
    This class provides an efficient inference wrapper that can store
    predictions and embeddings directly to memory-mapped arrays for
    large-scale evaluation.
    """
    
    def __init__(
        self,
        model: BertClassifier,
        embedding_storage: Optional[np.memmap] = None,
        prediction_storage: Optional[np.memmap] = None,
        feat_shrink: Optional[int] = None
    ):
        """
        Initialize the inference model.
        
        Args:
            model (BertClassifier): Trained BERT classifier
            embedding_storage (Optional[np.memmap]): Memory-mapped array for embeddings
            prediction_storage (Optional[np.memmap]): Memory-mapped array for predictions
            feat_shrink (Optional[int]): Feature dimension for embeddings
        """
        super(BertClassificationInferenceModel, self).__init__()
        
        self.model = model
        self.embedding_storage = embedding_storage
        self.prediction_storage = prediction_storage
        self.feat_shrink = feat_shrink or model.bert_hidden_size
        self.sample_count = 0
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass that stores results in memory-mapped arrays.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            **kwargs: Additional arguments
        
        Returns:
            torch.Tensor: Model logits
        """
        batch_size = input_ids.size(0)
        
        # Get predictions and features
        with torch.no_grad():
            logits, features = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_features=True
            )
            
            # Convert to numpy for storage
            logits_np = logits.cpu().numpy().astype(np.float16)
            features_np = features.cpu().numpy().astype(np.float16)
            
            # Store in memory-mapped arrays
            end_idx = self.sample_count + batch_size
            
            if self.prediction_storage is not None:
                self.prediction_storage[self.sample_count:end_idx] = logits_np
            
            if self.embedding_storage is not None:
                self.embedding_storage[self.sample_count:end_idx] = features_np
            
            self.sample_count += batch_size
        
        return logits


class AdvancedLanguageModelTrainer:
    """
    Advanced trainer for language models with comprehensive features.
    
    This class provides a sophisticated training pipeline for language models
    with support for various optimizations, evaluation strategies, and
    monitoring capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced language model trainer.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all parameters
        """
        self.config = config
        self.dataset_name = config.get('dataset', 'default')
        self.seed = config.get('seed', 42)
        
        # Model configuration
        model_config = config.get('model', {})
        self.model_config = ModelConfig(
            model_name=model_config.get('name', 'bert-base-uncased'),
            feat_shrink=model_config.get('feat_shrink', None),
            dropout=model_config.get('dropout', 0.1),
            attention_dropout=model_config.get('attention_dropout', 0.1),
            classifier_dropout=model_config.get('classifier_dropout', 0.1)
        )
        
        # Training configuration
        train_config = config.get('training', {})
        self.train_config = TrainingConfig(
            batch_size=train_config.get('batch_size', 16),
            learning_rate=train_config.get('learning_rate', 2e-5),
            weight_decay=train_config.get('weight_decay', 0.01),
            epochs=train_config.get('epochs', 10),
            warmup_epochs=train_config.get('warmup_epochs', 0.1),
            eval_patience=train_config.get('eval_patience', 500),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
            use_mixed_precision=train_config.get('use_mixed_precision', True)
        )
        
        # Setup paths
        self.use_gpt_str = "2" if config.get('use_gpt', False) else ""
        self.output_dir = Path(f'output/{self.dataset_name}{self.use_gpt_str}/{self.model_config.model_name}-seed{self.seed}')
        self.checkpoint_dir = Path(f'checkpoints/{self.dataset_name}{self.use_gpt_str}/{self.model_config.model_name}-seed{self.seed}')
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and data
        self._setup_model_and_data()
        
        # Training metrics
        self.training_history = defaultdict(list)
    
    def _setup_model_and_data(self):
        """Setup model, tokenizer, and datasets."""
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        bert_model = AutoModel.from_pretrained(self.model_config.model_name)
        
        # Create classifier
        self.model = BertClassifier(
            bert_model=bert_model,
            num_labels=self.model_config.num_labels,
            feat_shrink=self.model_config.feat_shrink,
            dropout=self.model_config.dropout,
            classifier_dropout=self.model_config.classifier_dropout
        )
        
        # Configure model dropout
        self.model.bert.config.hidden_dropout_prob = self.model_config.dropout
        self.model.bert.config.attention_probs_dropout_prob = self.model_config.attention_dropout
        
        # Log model information
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} trainable parameters")
    
    def create_datasets(
        self,
        texts: List[str],
        labels: List[int],
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train, validation, and test datasets.
        
        Args:
            texts (List[str]): List of text samples
            labels (List[int]): List of corresponding labels
            train_mask (torch.Tensor): Boolean mask for training samples
            val_mask (torch.Tensor): Boolean mask for validation samples
            test_mask (torch.Tensor): Boolean mask for test samples
        
        Returns:
            Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets
        """
        # Tokenize all texts
        tokenized_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset class
        class TextClassificationDataset(Dataset):
            def __init__(self, inputs, labels, mask):
                self.inputs = inputs
                self.labels = labels
                self.indices = mask.nonzero().squeeze().tolist()
                if isinstance(self.indices, int):
                    self.indices = [self.indices]
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                return {
                    'input_ids': self.inputs['input_ids'][real_idx],
                    'attention_mask': self.inputs['attention_mask'][real_idx],
                    'labels': torch.tensor(self.labels[real_idx], dtype=torch.long)
                }
        
        # Create datasets
        train_dataset = TextClassificationDataset(tokenized_inputs, labels, train_mask)
        val_dataset = TextClassificationDataset(tokenized_inputs, labels, val_mask)
        test_dataset = TextClassificationDataset(tokenized_inputs, labels, test_mask)
        
        logger.info(f"Datasets created - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        use_early_stopping: bool = True
    ) -> Dict[str, float]:
        """
        Train the model with advanced training strategies.
        
        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
            use_early_stopping (bool): Whether to use early stopping
        
        Returns:
            Dict[str, float]: Training metrics and results
        """
        logger.info("Starting model training")
        start_time = time.time()
        
        # Calculate training steps
        total_samples = len(train_dataset)
        effective_batch_size = self.train_config.batch_size * self.train_config.gradient_accumulation_steps
        steps_per_epoch = total_samples // effective_batch_size + (1 if total_samples % effective_batch_size else 0)
        total_steps = steps_per_epoch * self.train_config.epochs
        warmup_steps = int(self.train_config.warmup_epochs * total_steps)
        
        logger.info(f"Training configuration:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            do_train=True,
            do_eval=True,
            eval_steps=self.train_config.eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=self.train_config.eval_steps,
            save_strategy=self.train_config.save_strategy,
            learning_rate=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            max_grad_norm=self.train_config.max_grad_norm,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='eval_accuracy',
            greater_is_better=True,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            per_device_train_batch_size=self.train_config.batch_size,
            per_device_eval_batch_size=self.train_config.batch_size * 2,
            warmup_steps=warmup_steps,
            num_train_epochs=self.train_config.epochs,
            dataloader_num_workers=4,
            fp16=self.train_config.use_mixed_precision,
            dataloader_drop_last=True,
            logging_steps=self.train_config.logging_steps,
            report_to=None,  # Disable wandb/tensorboard logging
        )
        
        # Setup callbacks
        callbacks = []
        if use_early_stopping:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
            callbacks.append(early_stopping)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_comprehensive_metrics,
            callbacks=callbacks
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save model and training state
        self.trainer.save_model()
        self.trainer.save_state()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Return training metrics
        return {
            'train_loss': train_result.training_loss,
            'train_steps': train_result.global_step,
            'training_time': training_time
        }
    
    def evaluate_and_save(
        self,
        full_dataset: Dataset,
        save_embeddings: bool = True,
        save_predictions: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation with optional embedding and prediction saving.
        
        Args:
            full_dataset (Dataset): Complete dataset for evaluation
            save_embeddings (bool): Whether to save embeddings to disk
            save_predictions (bool): Whether to save predictions to disk
        
        Returns:
            Dict[str, float]: Comprehensive evaluation metrics
        """
        logger.info("Starting comprehensive evaluation")
        
        total_samples = len(full_dataset)
        
        # Setup memory-mapped storage if requested
        embedding_storage = None
        prediction_storage = None
        
        if save_embeddings:
            emb_path = self.output_dir / "embeddings.dat"
            embedding_storage = np.memmap(
                emb_path,
                dtype=np.float16,
                mode='w+',
                shape=(total_samples, self.model_config.feat_shrink or self.model.bert_hidden_size)
            )
        
        if save_predictions:
            pred_path = self.output_dir / "predictions.dat"
            prediction_storage = np.memmap(
                pred_path,
                dtype=np.float16,
                mode='w+',
                shape=(total_samples, self.model_config.num_labels)
            )
        
        # Create inference model
        inference_model = BertClassificationInferenceModel(
            model=self.model,
            embedding_storage=embedding_storage,
            prediction_storage=prediction_storage,
            feat_shrink=self.model_config.feat_shrink
        )
        
        # Setup inference arguments
        inference_args = TrainingArguments(
            output_dir=str(self.output_dir),
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.train_config.batch_size * 4,
            dataloader_drop_last=False,
            dataloader_num_workers=4,
            fp16_full_eval=self.train_config.use_mixed_precision
        )
        
        # Create inference trainer
        inference_trainer = Trainer(
            model=inference_model,
            args=inference_args
        )
        
        # Run inference
        predictions = inference_trainer.predict(full_dataset)
        
        # Extract labels for evaluation
        all_labels = [sample['labels'].item() for sample in full_dataset]
        
        # Compute comprehensive metrics
        metrics = compute_comprehensive_metrics((predictions.predictions, all_labels))
        
        # Log results
        logger.info("Evaluation Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        # Save metrics
        metrics_path = self.output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (Optional[str]): Path to checkpoint. If None, uses default path
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "model.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dict containing training configuration and results
        """
        summary = {
            'dataset': self.dataset_name,
            'model_name': self.model_config.model_name,
            'seed': self.seed,
            'model_config': self.model_config.__dict__,
            'training_config': self.train_config.__dict__,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        return summary


# Legacy compatibility functions

def compute_metrics(predictions_and_labels: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Legacy compute_metrics function for backward compatibility."""
    return {"accuracy": compute_comprehensive_metrics(predictions_and_labels)["accuracy"]}


class LMTrainer:
    """Legacy LMTrainer class for backward compatibility."""
    
    def __init__(self, cfg):
        """Initialize with legacy config format."""
        warnings.warn("LMTrainer is deprecated. Use AdvancedLanguageModelTrainer instead.", 
                     DeprecationWarning, stacklevel=2)
        
        # Convert legacy config to new format
        modern_config = {
            'dataset': getattr(cfg, 'dataset', 'default'),
            'seed': getattr(cfg, 'seed', 42),
            'model': {
                'name': getattr(cfg.lm.model, 'name', 'bert-base-uncased'),
                'feat_shrink': getattr(cfg.lm.model, 'feat_shrink', None),
                'dropout': getattr(cfg.lm.train, 'dropout', 0.1),
                'attention_dropout': getattr(cfg.lm.train, 'att_dropout', 0.1),
                'classifier_dropout': getattr(cfg.lm.train, 'cla_dropout', 0.1)
            },
            'training': {
                'batch_size': getattr(cfg.lm.train, 'batch_size', 16),
                'learning_rate': getattr(cfg.lm.train, 'lr', 2e-5),
                'weight_decay': getattr(cfg.lm.train, 'weight_decay', 0.01),
                'epochs': getattr(cfg.lm.train, 'epochs', 10),
                'warmup_epochs': getattr(cfg.lm.train, 'warmup_epochs', 0.1),
                'eval_patience': getattr(cfg.lm.train, 'eval_patience', 500),
                'gradient_accumulation_steps': getattr(cfg.lm.train, 'grad_acc_steps', 1),
                'use_mixed_precision': True
            },
            'use_gpt': getattr(cfg.lm.train, 'use_gpt', False)
        }
        
        self.trainer = AdvancedLanguageModelTrainer(modern_config)
    
    def train(self):
        """Legacy train method."""
        return self.trainer.train()
    
    def eval_and_save(self):
        """Legacy eval_and_save method."""
        return self.trainer.evaluate_and_save()
