#!/usr/bin/env python3
"""
Main training script for Hanoi Graph Transformer.
Trains the model on Tower of Hanoi puzzles with variable disk counts.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
from pathlib import Path

# Import our modules
from GraphModel import HanoiGraphTransformer
from HanoiDataset import load_dataset, create_pytorch_dataset, hanoi_collate_fn
from model_trainer import HanoiTrainer


def create_train_val_split(data, val_split=0.2, random_state=42):
    """Split dataset into train/validation sets."""
    # Split by N values to ensure both sets have similar distribution
    train_data = []
    val_data = []
    
    # Group by N value
    data_by_N = {}
    for sample in data:
        N = sample['N']
        if N not in data_by_N:
            data_by_N[N] = []
        data_by_N[N].append(sample)
    
    # Split each N group separately
    for N, samples in data_by_N.items():
        train_samples, val_samples = train_test_split(
            samples, test_size=val_split, random_state=random_state
        )
        train_data.extend(train_samples)
        val_data.extend(val_samples)
    
    return train_data, val_data


def main():
    """Main training function."""
    print("="*70)
    print("Hanoi Graph Transformer Training")
    print("="*70)
    
    # Configuration
    config = {
        # Training Data
        'N_values': list(range(3, 8)),  # [3, 4, 5, 6, 7, 8, 9, 10]
        'samples_per_N': 3000,
        'noise_rate': 0.1,
        'val_split': 0.2,
        'variable_length': True,
        
        # Model - Support up to N=15 for generalization testing
        'N_max': 24,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0,
        'K_fourier': 12,
        'use_structural_bias': False,
        
        # Training
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5, 
        'grad_clip': 1.0,
        'scheduler': None, # 'cosine', 'step', or None
        'min_lr': 1e-6,
        
        # Loss
        'entropy_weight': 0,
        'label_smoothing': 0.05,
        
        # Logging
        'save_every': 25,
        'checkpoint_dir': './checkpoints',
        'num_workers': 4,
    }
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n" + "="*70)
    print("Loading Dataset")
    print("="*70)
    
    dataset_path = './hanoi_data/hanoi_dataset_N3_to_10_bfs.pkl'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please generate the dataset first by running HanoiDataset.py")
        return
    
    loaded_data, loaded_metadata = load_dataset(dataset_path)
    print(f"Loaded {len(loaded_data):,} samples")
    print(f"N range: {loaded_metadata['N_range']}")
    print(f"Method: {loaded_metadata['method']}")
    
    # Create train/val split
    print("\n" + "="*70)
    print("Creating Train/Validation Split")
    print("="*70)
    
    train_data, val_data = create_train_val_split(
        loaded_data, 
        val_split=config['val_split'],
        random_state=42
    )
    
    print(f"Train samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    
    # Create PyTorch datasets
    train_dataset = create_pytorch_dataset(train_data, N_filter=config['N_values'])
    val_dataset = create_pytorch_dataset(val_data, N_filter=config['N_values'])
    
    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    model = HanoiGraphTransformer(
        N_max=config['N_max'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        K_fourier=config['K_fourier'],
        use_structural_bias=config['use_structural_bias']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model architecture:")
    print(f"  - N_max: {config['N_max']}")
    print(f"  - d_model: {config['d_model']}")
    print(f"  - n_heads: {config['n_heads']}")
    print(f"  - n_layers: {config['n_layers']}")
    print(f"  - d_ff: {config['d_ff']}")
    print(f"  - K_fourier: {config['K_fourier']}")
    print(f"  - use_structural_bias: {config['use_structural_bias']}")
    
    # Create trainer
    print("\n" + "="*70)
    print("Creating Trainer")
    print("="*70)
    
    trainer = HanoiTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )
    
    print(f"Training configuration:")
    print(f"  - Epochs: {config['num_epochs']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Weight decay: {config['weight_decay']}")
    print(f"  - Scheduler: {config['scheduler']}")
    print(f"  - Entropy weight: {config['entropy_weight']}")
    print(f"  - Label smoothing: {config['label_smoothing']}")
    
    # Train the model
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Best validation accuracy: {trainer.best_val_accuracy:.3f}")
    
    # Save model
    print("\n" + "="*70)
    print("Saving Model")
    print("="*70)
    
    model_path = './trained_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_accuracy': trainer.best_val_accuracy,
        'best_val_loss': trainer.best_val_loss
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best validation accuracy: {trainer.best_val_accuracy:.3f}")
    print(f"Training time: {training_time:.1f} seconds")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
