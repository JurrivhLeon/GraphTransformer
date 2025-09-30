import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from HanoiDataset import HanoiDataset, create_pytorch_dataset, hanoi_collate_fn, load_dataset
from GraphModel import HanoiGraphTransformer
from model_trainer import HanoiTrainer
from main import create_train_val_split
from torch.utils.data import DataLoader
import os

def debug_model_training(model, train_loader, val_loader, device='cuda'):
    """Comprehensive debugging for Hanoi model training issues."""
    
    print("="*60)
    print("HANOI MODEL TRAINING DIAGNOSTICS")
    print("="*60)
    
    # 1. Check data quality
    print("\n1. DATA QUALITY CHECK")
    print("-" * 40)
    
    batch = next(iter(train_loader))
    states = batch['state']
    targets = batch['noisy_target']
    teacher_moves = batch['teacher_move']
    
    print(f"Batch shape: states {states.shape}, targets {targets.shape}")
    print(f"State values range: [{states.min()}, {states.max()}]")
    print(f"Target values range: [{targets.min().item():.3f}, {targets.max().item():.3f}]")
    
    # Check target distribution
    num_legal_per_sample = (targets > 0).sum(dim=(1,2)).float()
    print(f"Legal moves per sample: mean={num_legal_per_sample.mean():.2f}, "
          f"min={num_legal_per_sample.min():.0f}, max={num_legal_per_sample.max():.0f}")
    
    # Check if targets sum to 1
    target_sums = targets.sum(dim=(1,2))
    print(f"Target probability sums: mean={target_sums.mean():.3f}, "
          f"min={target_sums.min():.3f}, max={target_sums.max():.3f}")
    if not torch.allclose(target_sums, torch.ones_like(target_sums), atol=1e-5):
        print("⚠️  WARNING: Targets don't sum to 1.0!")
    
    # Check teacher move consistency
    for i in range(min(3, states.shape[0])):
        tm = teacher_moves[i]
        target_at_tm = targets[i, tm[0], tm[1]]
        print(f"Sample {i}: teacher_move=({tm[0]}, {tm[1]}), prob={target_at_tm:.3f}")
    
    # 2. Check model output
    print("\n2. MODEL OUTPUT CHECK")
    print("-" * 40)
    
    model.eval()
    with torch.no_grad():
        states = states.to(device)
        lengths = batch['lengths'].to(device)
        logits = model(states, lengths)
        
        print(f"Logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        print(f"Logits mean: {logits.mean().item():.3f}, std: {logits.std().item():.3f}")
        
        # Check if model outputs are diverse
        logits_flat = logits.view(-1, 9)
        logits_std_per_sample = logits_flat.std(dim=1)
        print(f"Logits std per sample: mean={logits_std_per_sample.mean():.3f}")
        if logits_std_per_sample.mean() < 0.1:
            print("⚠️  WARNING: Logits have very low variance - model might not be learning!")
        
        # Check predictions
        probs = F.softmax(logits_flat, dim=-1).view(-1, 3, 3)
        pred_moves = torch.argmax(logits_flat, dim=-1)
        pred_from = pred_moves // 3
        pred_to = pred_moves % 3
        
        # Compare with teacher
        teacher_flat = teacher_moves[:, 0] * 3 + teacher_moves[:, 1]
        accuracy = (pred_moves == teacher_flat).float().mean()
        print(f"Batch accuracy: {accuracy.item():.3f}")
        
        # Show some predictions
        for i in range(min(3, states.shape[0])):
            tm = teacher_moves[i]
            pm_from, pm_to = pred_from[i].item(), pred_to[i].item()
            teacher_prob = probs[i, tm[0], tm[1]].item()
            pred_prob = probs[i, pm_from, pm_to].item()
            print(f"Sample {i}: teacher=({tm[0]}, {tm[1]})[{teacher_prob:.3f}], "
                  f"pred=({pm_from}, {pm_to})[{pred_prob:.3f}]")
    
    # 3. Check loss computation
    print("\n3. LOSS COMPUTATION CHECK")
    print("-" * 40)
    
    targets = targets.to(device)
    flat_logits = logits.view(-1, 9)
    flat_targets = targets.view(-1, 9)
    
    # Method 1: Cross-entropy (what we should use)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    loss_ce = -(flat_targets * log_probs).sum(dim=-1).mean()
    print(f"Cross-entropy loss: {loss_ce.item():.4f}")
    
    # Check if loss is in reasonable range
    # Random guessing with ~3 legal moves: -log(1/3) ≈ 1.099
    print(f"Random baseline loss: ~1.099")
    if loss_ce.item() > 1.5:
        print("⚠️  WARNING: Loss is higher than random - model is worse than random!")
    
    # 4. Check gradient flow
    print("\n4. GRADIENT FLOW CHECK")
    print("-" * 40)
    
    model.train()
    model.zero_grad()
    
    # Forward and backward
    logits = model(states, lengths)
    loss = -(flat_targets * F.log_softmax(logits.view(-1, 9), dim=-1)).sum(dim=-1).mean()
    loss.backward()
    
    # Check gradient statistics
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1
            
            if param_norm < 1e-7:
                zero_grad_count += 1
        else:
            print(f"⚠️  No gradient for: {name}")
    
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Parameters with gradients: {param_count}")
    print(f"Parameters with ~zero gradients: {zero_grad_count}")
    
    if total_norm < 1e-6:
        print("⚠️  WARNING: Gradients are vanishing!")
    elif total_norm > 100:
        print("⚠️  WARNING: Gradients might be exploding!")
    
    # Show gradient norms for key layers
    print("\nGradient norms by layer:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            if 'weight' in name or 'bias' in name:
                layer_name = name.split('.')[0:3]
                print(f"  {'.'.join(layer_name)}: {grad_norm:.6f}")
    
    # 5. Check attention patterns (if using structural bias)
    print("\n5. ATTENTION PATTERN CHECK")
    print("-" * 40)
    
    if hasattr(model, 'use_structural_bias') and model.use_structural_bias:
        # Get attention from first layer
        model.eval()
        with torch.no_grad():
            # Hook to capture attention
            attention_weights = []
            
            def hook_fn(module, input, output):
                # This is a simplified check
                pass
            
            print("Structural bias enabled")
            # Would need to add hooks to actually inspect attention
    
    # 6. Learning rate check
    print("\n6. TRAINING CONFIGURATION")
    print("-" * 40)
    print("Recommended checks:")
    print("  - Learning rate: 1e-4 to 5e-4 (AdamW)")
    print("  - Batch size: 16-64")
    print("  - Gradient clipping: 1.0")
    print("  - Noise rate: 0.1")
    print("  - Model size: d_model=128-256")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


def visualize_predictions(model, val_loader, device='cuda', num_samples=50):
    """Visualize model predictions vs ground truth."""
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_accuracies = []
    
    with torch.no_grad():
        for batch in val_loader:
            states = batch['state'].to(device)
            lengths = batch['lengths'].to(device)
            targets = batch['noisy_target'].to(device)
            teacher_moves = batch['teacher_move']
            
            logits = model(states, lengths)
            
            # Get predictions
            logits_flat = logits.view(-1, 9)
            pred_moves = torch.argmax(logits_flat, dim=-1)
            teacher_flat = teacher_moves[:, 0] * 3 + teacher_moves[:, 1]
            
            correct = (pred_moves == teacher_flat.to(device)).cpu().numpy()
            all_accuracies.extend(correct.tolist())
            
            if len(all_accuracies) >= num_samples:
                break
    
    accuracy = np.mean(all_accuracies[:num_samples])
    print(f"\nValidation Accuracy: {accuracy:.3f}")
    print(f"Correct predictions: {sum(all_accuracies[:num_samples])}/{num_samples}")
    
    # Distribution analysis
    from collections import Counter
    counter = Counter(all_accuracies[:num_samples])
    print(f"Correct: {counter[True]}, Incorrect: {counter[False]}")
    
    return accuracy


def test_overfitting_capability(model, train_loader, device='cuda', num_epochs=50):
    """Test if model can overfit on small dataset (sanity check)."""
    
    print("\n" + "="*60)
    print("OVERFITTING SANITY CHECK")
    print("="*60)
    print("Testing if model can memorize a small batch...")
    
    # Get one small batch
    batch = next(iter(train_loader))
    states = batch['state'][:8].to(device)
    lengths = batch['lengths'][:8].to(device)
    targets = batch['noisy_target'][:8].to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        logits = model(states, lengths)
        
        # Compute loss
        flat_logits = logits.view(-1, 9)
        flat_targets = targets.view(-1, 9)
        log_probs = F.log_softmax(flat_logits, dim=-1)
        loss = -(flat_targets * log_probs).sum(dim=-1).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            pred_moves = torch.argmax(flat_logits, dim=-1)
            teacher_flat = torch.argmax(flat_targets, dim=-1)
            acc = (pred_moves == teacher_flat).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.3f}")
    
    final_acc = accuracies[-1]
    print(f"\nFinal accuracy on 8 samples: {final_acc:.3f}")
    
    if final_acc < 0.8:
        print("⚠️  CRITICAL: Model cannot overfit even on 8 samples!")
        print("   This suggests a fundamental issue with:")
        print("   - Model architecture")
        print("   - Loss computation")
        print("   - Data format")
    else:
        print("✓ Model CAN learn (overfitting works)")
        print("  Issue is likely:")
        print("  - Learning rate too low")
        print("  - Not enough training")
        print("  - Dataset too noisy")
    
    return losses, accuracies


# Quick usage example
if __name__ == "__main__":
    from hanoi_debug import debug_model_training, test_overfitting_capability
    # Comprehensive diagnostics
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
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 3,
        'd_ff': 1024,
        'dropout': 0.1,
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
    
    train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=hanoi_collate_fn,
            num_workers=config.get('num_workers', 0)
    )

    val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=hanoi_collate_fn,
            num_workers=config.get('num_workers', 0)
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
    debug_model_training(model, train_loader, val_loader, device)
    
    # Sanity check: Can the model even overfit?
    # test_overfitting_capability(model, train_loader, device)