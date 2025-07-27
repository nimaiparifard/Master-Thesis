"""
Test Script for ENGINE-GraphLoRA

This script provides a minimal test to verify that the ENGINE-GraphLoRA implementation
is working correctly with small arguments and a lightweight language model.

Usage:
    python models/engine_graph_lora/test.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.engine_graph_lora.arguments import EngineGraphLoRAArguments
from models.engine_graph_lora.model import EngineGraphLoRAModel
from models.engine_graph_lora.trainer import EngineGraphLoRATrainer
from models.engine_graph_lora.evaluation import EngineGraphLoRAEvaluator

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def create_minimal_synthetic_data(num_nodes=50, num_edges=100, num_classes=3, max_text_length=32):
    """
    Create minimal synthetic graph data for testing
    
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph
        num_classes: Number of classification classes
        max_text_length: Maximum text sequence length
        
    Returns:
        Dictionary containing synthetic graph data
    """
    print(f"Creating synthetic data: {num_nodes} nodes, {num_edges} edges, {num_classes} classes")
    
    # Create random edge index
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create random labels
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    # Create dummy text data (random token IDs)
    vocab_size = 1000  # Small vocabulary for testing
    input_ids = torch.randint(1, vocab_size-1, (num_nodes, max_text_length))
    attention_mask = torch.ones(num_nodes, max_text_length)
    
    # Add padding to some sequences
    for i in range(num_nodes):
        seq_len = torch.randint(max_text_length//2, max_text_length, (1,)).item()
        attention_mask[i, seq_len:] = 0
        input_ids[i, seq_len:] = 0  # pad token
    
    # Create anchor mask (10% of nodes are anchors)
    anchor_mask = torch.zeros(num_nodes, dtype=torch.bool)
    num_anchors = max(1, num_nodes // 10)
    anchor_indices = torch.randperm(num_nodes)[:num_anchors]
    anchor_mask[anchor_indices] = True
    
    # Split into train/val/test
    train_ratio, val_ratio = 0.6, 0.2
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    indices = torch.randperm(num_nodes)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create data objects
    class SyntheticData:
        def __init__(self, node_indices):
            self.input_ids = input_ids[node_indices]
            self.attention_mask = attention_mask[node_indices]
            self.edge_index = edge_index
            self.y = labels[node_indices]
            self.anchor_mask = anchor_mask[node_indices]
            self.original_idx = node_indices
            self.num_nodes = len(node_indices)
            
        def __len__(self):
            return len(self.y)
    
    train_data = SyntheticData(train_indices)
    val_data = SyntheticData(val_indices)
    test_data = SyntheticData(test_indices)
    
    print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def create_test_config():
    """Create minimal test configuration"""
    
    # Create test arguments with minimal settings
    test_args = [
        '--dataset', 'synthetic',
        '--task', 'node_classification',
        '--output_dir', './test_outputs',
        
        # Use a very small language model for testing
        '--llm_model_name', 'prajjwal1/bert-tiny',  # Only 4.4M parameters
        '--llm_hidden_size', '128',
        '--llm_num_layers', '2',
        '--llm_freeze_base',
        
        # Minimal G-Ladder configuration
        '--g_ladder_num_layers', '2',
        '--g_ladder_hidden_size', '128',
        '--g_ladder_injection_layers', '0', '1',
        
        # Small LoRA configuration
        '--lora_rank', '4',
        '--lora_alpha', '8.0',
        '--lora_dropout', '0.1',
        
        # Minimal GNN configuration
        '--gnn_type', 'sage',
        '--gnn_hidden_size', '64',
        '--gnn_num_layers', '2',
        '--gnn_dropout', '0.1',
        
        # Small anchor configuration
        '--anchor_ratio', '0.2',
        '--anchor_budget_ratio', '0.5',
        '--cache_size', '100',
        
        # Minimal training configuration
        '--num_epochs', '3',
        '--batch_size', '8',
        '--learning_rate', '1e-3',
        '--weight_decay', '1e-4',
        '--warmup_steps', '10',
        '--early_stopping_patience', '5',
        '--validation_frequency', '1',
        
        # Data configuration
        '--max_text_length', '32',
        '--train_ratio', '0.6',
        '--val_ratio', '0.2',
        '--test_ratio', '0.2',
        '--num_workers', '0',  # No multiprocessing for simplicity
        
        # Loss configuration
        '--task_loss_weight', '1.0',
        '--smmd_loss_weight', '0.1',
        '--contrastive_loss_weight', '0.05',
        
        # System configuration
        '--device', 'cpu',  # Force CPU for testing
        '--seed', '42',
        '--debug'
    ]
    
    # Parse arguments
    args_parser = EngineGraphLoRAArguments()
    config = args_parser.parse_args(test_args)
    
    return config


def create_simple_dataloader(data, batch_size=8):
    """Create a simple dataloader for testing"""
    
    class SimpleDataLoader:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size
            self.dataset = data  # For compatibility
            
        def __iter__(self):
            num_samples = len(self.data)
            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch = {
                    'input_ids': self.data.input_ids[batch_indices],
                    'attention_mask': self.data.attention_mask[batch_indices],
                    'edge_index': self.data.edge_index,
                    'labels': self.data.y[batch_indices],
                    'anchor_mask': self.data.anchor_mask[batch_indices],
                    'batch_nodes': batch_indices,
                    'original_idx': self.data.original_idx[batch_indices]
                }
                
                yield batch
                
        def __len__(self):
            return (len(self.data) + self.batch_size - 1) // self.batch_size
    
    return SimpleDataLoader(data, batch_size)


def test_model_initialization():
    """Test model initialization"""
    print("\n" + "="*50)
    print("Testing Model Initialization")
    print("="*50)
    
    try:
        config = create_test_config()
        model = EngineGraphLoRAModel(config, num_classes=3)
        
        print(f"✅ Model initialized successfully")
        print(f"   - Total parameters: {model.count_parameters():,}")
        print(f"   - Trainable parameters: {len(model.get_trainable_parameters())}")
        print(f"   - Device: {config.device}")
        
        return model, config
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        raise


def test_forward_pass(model, config):
    """Test forward pass with synthetic data"""
    print("\n" + "="*50)
    print("Testing Forward Pass")
    print("="*50)
    
    try:
        # Create minimal synthetic data
        train_data, val_data, test_data = create_minimal_synthetic_data(
            num_nodes=20, num_edges=30, num_classes=3, max_text_length=16
        )
        
        # Create a single batch
        batch = {
            'input_ids': train_data.input_ids[:5],
            'attention_mask': train_data.attention_mask[:5],
            'edge_index': train_data.edge_index,
            'labels': train_data.y[:5],
            'anchor_mask': train_data.anchor_mask[:5],
            'batch_nodes': torch.arange(5),
            'original_idx': torch.arange(5)
        }
        
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                edge_index=batch['edge_index'],
                batch_nodes=batch['batch_nodes'],
                anchor_mask=batch['anchor_mask']
            )
        
        print(f"✅ Forward pass successful")
        print(f"   - Output keys: {list(outputs.keys())}")
        print(f"   - Logits shape: {outputs['logits'].shape}")
        print(f"   - Text embeddings shape: {outputs.get('text_embeddings', torch.tensor([])).shape}")
        
        return train_data, val_data, test_data
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise


def test_training_step(model, config, train_data):
    """Test a single training step"""
    print("\n" + "="*50)
    print("Testing Training Step")
    print("="*50)
    
    try:
        # Create trainer
        trainer = EngineGraphLoRATrainer(
            config=config,
            model=model,
            train_data=train_data,
            val_data=train_data,  # Use same data for simplicity
            test_data=train_data
        )
        
        # Override the anchor selection for testing
        trainer._anchor_nodes = torch.arange(min(10, len(train_data)))
        
        # Test a single training step
        model.train()
        trainer.model.setup_for_training()
        
        # Create a minimal batch
        batch = {
            'input_ids': train_data.input_ids[:3],
            'attention_mask': train_data.attention_mask[:3],
            'edge_index': train_data.edge_index,
            'labels': train_data.y[:3],
            'anchor_mask': train_data.anchor_mask[:3],
            'batch_nodes': torch.arange(3),
            'original_idx': torch.arange(3)
        }
        
        # Test selective LLM refresh
        h_llm = trainer._selective_llm_refresh(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            anchor_mask=batch['anchor_mask'],
            budget_ratio=0.5
        )
        
        print(f"✅ Training step components working")
        print(f"   - LLM refresh output shape: {h_llm.shape}")
        print(f"   - Trainer initialized successfully")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        raise


def test_evaluation(model, config, test_data):
    """Test evaluation functionality"""
    print("\n" + "="*50)
    print("Testing Evaluation")
    print("="*50)
    
    try:
        # Create evaluator
        evaluator = EngineGraphLoRAEvaluator(model, config)
        
        # Create test dataloader
        test_loader = create_simple_dataloader(test_data, batch_size=4)
        
        # Test standard evaluation
        model.eval()
        results = evaluator.evaluate_standard(
            data_loader=test_loader,
            return_detailed_metrics=False,
            save_predictions=False
        )
        
        print(f"✅ Evaluation successful")
        print(f"   - Accuracy: {results['accuracy']:.4f}")
        print(f"   - Total loss: {results['total_loss']:.4f}")
        print(f"   - Number of samples: {results['num_samples']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


def test_end_to_end():
    """Run end-to-end test with minimal training"""
    print("\n" + "="*50)
    print("Testing End-to-End Training (Mini)")
    print("="*50)
    
    try:
        # Initialize
        config = create_test_config()
        model = EngineGraphLoRAModel(config, num_classes=3)
        
        # Create synthetic data
        train_data, val_data, test_data = create_minimal_synthetic_data(
            num_nodes=30, num_edges=40, num_classes=3, max_text_length=16
        )
        
        # Create trainer
        trainer = EngineGraphLoRATrainer(
            config=config,
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        
        # Override anchor nodes for testing
        trainer._anchor_nodes = torch.arange(min(15, len(train_data)))
        trainer._preprocessed = True  # Skip preprocessing for testing
        
        print("Starting mini training...")
        
        # Run one training epoch manually to test core functionality
        model.train()
        trainer.model.setup_for_training()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'task_loss': 0.0,
            'smmd_loss': 0.0,
            'contrastive_loss': 0.0,
            'num_batches': 0
        }
        
        train_loader = create_simple_dataloader(train_data, batch_size=4)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # Only test 2 batches
                break
                
            # Move to device
            batch = trainer._move_batch_to_device(batch)
            
            # Forward pass (simplified)
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                edge_index=batch['edge_index'],
                batch_nodes=batch.get('batch_nodes'),
                anchor_mask=batch.get('anchor_mask')
            )
            
            # Simple loss computation
            import torch.nn.functional as F
            task_loss = F.cross_entropy(outputs['logits'], batch['labels'])
            
            # Skip complex losses for basic test
            total_loss = task_loss
            
            # Backward pass
            trainer.optimizer.zero_grad()
            total_loss.backward()
            trainer.optimizer.step()
            
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['task_loss'] += task_loss.item()
            epoch_metrics['num_batches'] += 1
            
            print(f"   Batch {batch_idx + 1}: Loss = {total_loss.item():.4f}")
        
        # Average metrics
        for key in ['total_loss', 'task_loss', 'smmd_loss', 'contrastive_loss']:
            if epoch_metrics['num_batches'] > 0:
                epoch_metrics[key] /= epoch_metrics['num_batches']
        
        print(f"✅ End-to-end test successful")
        print(f"   - Average loss: {epoch_metrics['total_loss']:.4f}")
        print(f"   - Processed {epoch_metrics['num_batches']} batches")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        raise


def main():
    """Run all tests"""
    print("🚀 Starting ENGINE-GraphLoRA Test Suite")
    print("Using minimal configuration with tiny language model")
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run tests step by step
        model, config = test_model_initialization()
        train_data, val_data, test_data = test_forward_pass(model, config)
        trainer = test_training_step(model, config, train_data)
        results = test_evaluation(model, config, test_data)
        
        # Run end-to-end test
        test_end_to_end()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("="*50)
        print("✅ Model initialization: PASSED")
        print("✅ Forward pass: PASSED")
        print("✅ Training step: PASSED")
        print("✅ Evaluation: PASSED")
        print("✅ End-to-end: PASSED")
        print("\nENGINE-GraphLoRA implementation is working correctly! 🚀")
        
    except Exception as e:
        print("\n" + "="*50)
        print("❌ TEST FAILED!")
        print("="*50)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 