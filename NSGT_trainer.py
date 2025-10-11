import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
from collections import defaultdict
import time
import json
from tqdm import tqdm

# Import our modules
from NSGT import HanoiNextStateTransformer
from HanoiDataset import HanoiDatasetGenerator, HanoiDataset, TowerOfHanoiState, hanoi_next_state_collate_fn


class HanoiTrainer:
    """Trainer for Tower of Hanoi Graph Transformer."""
    def __init__(self, 
                 model: HanoiNextStateTransformer,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 config: Dict,
                 device: str = 'cuda'):
        
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Datasets and loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=hanoi_next_state_collate_fn,
            num_workers=config.get('num_workers', 0)
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=hanoi_next_state_collate_fn,
            num_workers=config.get('num_workers', 0)
        )
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4),
            betas=config.get('betas', (0.9, 0.999))
        )
        
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config['num_epochs'],
                eta_min=config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config.get('patience', 10),
                factor=config.get('lr_factor', 0.5)
            )
        else:
            self.scheduler = None
        
        # Loss configuration
        self.entropy_weight = config.get('entropy_weight', 0.0)
        self.label_smoothing = config.get('label_smoothing', 0.0)
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)

    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, state_info: Dict, next_moves: torch.Tensor, teacher_moves: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute training loss with optional entropy regularization."""
        batch_size, N_max, n_rods = logits.shape
        
        # Get padding mask from state_info
        padding_mask = state_info.get('padding_mask')  # (B, N_max)
        
        # Apply padding mask to targets
        if padding_mask is not None:
            valid_mask = padding_mask.unsqueeze(-1)  # (B, N_max, 1)
            targets = targets * valid_mask
        
        # Standard cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)  # (B, N_max, 3)
        loss = -torch.sum(targets * log_probs, dim=-1)  # (B, N_max)
        
        # Apply padding mask to loss
        if padding_mask is not None:
            loss = loss * padding_mask.float()
            loss = loss.sum(dim=-1) / (padding_mask.sum(dim=-1).float() + 1e-8)
        else:
            loss = loss.mean(dim=-1)  # (B,)
        
        # Entropy regularization
        entropy_loss = 0.0
        if self.entropy_weight > 0:
            probs = F.softmax(logits, dim=-1)  # (B, N_max, 3)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B, N_max)
            
            if padding_mask is not None:
                entropy = entropy * padding_mask.float()
                entropy = entropy.sum(dim=-1) / (padding_mask.sum(dim=-1).float() + 1e-8)
            else:
                entropy = entropy.mean(dim=-1)
            
            entropy_loss = -self.entropy_weight * entropy.mean()
        
        total_loss = loss.mean() + entropy_loss
        
        # Compute accuracy (correct teacher move prediction)
        with torch.no_grad():
            correct = (next_moves == teacher_moves).all(dim=-1).float()  # (B,)
            accuracy = correct.mean()
        
        metrics = {
            'cross_entropy': loss.mean().item(),
            'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
            'accuracy': accuracy.item()
        }
        
        return total_loss, metrics
    
    def train_step(self, batch: Union[Dict, List]) -> Dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        total_metrics = defaultdict(float)
        num_samples = 0
        
        # Standard batch processing
        states = batch['state'].to(self.device)
        lengths = batch.get('lengths')
        if lengths is not None:
            lengths = lengths.to(self.device)
        targets = batch['noisy_target'].to(self.device)
            
        # Forward pass
        logits, next_moves, state_info = self.model(batch)
        teacher_moves = batch['teacher_move'].to(self.device)
            
        # Compute loss
        total_loss, total_metrics = self.compute_loss(logits, targets, state_info, next_moves, teacher_moves)
        total_loss.backward()
        num_samples = states.shape[0]
        
        # Gradient clipping
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
        
        self.optimizer.step()
        
        # Metrics (already averaged inside compute_loss)
        metrics_out = dict(total_metrics)
        metrics_out['loss'] = total_loss.item()
        
        return metrics_out
    
    def validate(self) -> Dict:
        """Validation loop."""
        self.model.eval()
        
        total_loss = 0.0
        total_metrics = defaultdict(float)
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating', leave=False):
                # Standard validation
                states = batch['state'].to(self.device)
                lengths = batch.get('lengths')
                if lengths is not None:
                    lengths = lengths.to(self.device)
                targets = batch['noisy_target'].to(self.device)
                teacher_moves = batch['teacher_move'].to(self.device)
                logits, next_moves, state_info = self.model(batch)
                loss, metrics = self.compute_loss(logits, targets, state_info, next_moves, teacher_moves)
                total_loss += loss.item() * states.shape[0]
                num_samples += states.shape[0]
                    
                for k, v in metrics.items():
                    total_metrics[k] += v * states.shape[0]
        
        # Average metrics
        avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}
        avg_metrics['loss'] = total_loss / num_samples
        
        return avg_metrics
    
    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        for batch in pbar:
            metrics = self.train_step(batch)
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}"
            })
        
        # Average epoch metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        return avg_metrics
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.train_history = defaultdict(list, checkpoint['train_history'])
        self.val_history = defaultdict(list, checkpoint['val_history'])
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Store metrics
            for k, v in train_metrics.items():
                self.train_history[k].append(v)
            for k, v in val_metrics.items():
                self.val_history[k].append(v)
            
            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['accuracy']
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}")
            print(f"  Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}")
            if is_best:
                print(f"  *** New best accuracy: {self.best_val_accuracy:.3f} ***")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                checkpoint_path = os.path.join(
                    self.config.get('checkpoint_dir', './checkpoints'),
                    f'checkpoint_epoch_{epoch+1}.pt'
                )
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                self.save_checkpoint(checkpoint_path, is_best)
        
        print(f"Training completed! Best validation accuracy: {self.best_val_accuracy:.3f}")


'''def evaluate_generalization(model: HanoiGraphTransformer, 
                          test_N_values: List[int], 
                          samples_per_N: int = 500,
                          device: str = 'cuda',
                          max_solve_steps: int = 2000) -> Dict:
    """Evaluate model generalization on different puzzle sizes."""
    model.eval()
    # solver = OptimalSolver()
    results = {}
    
    print("=== Generalization Evaluation ===")
    
    for N in test_N_values:
        print(f"\nTesting on N={N} disks...")
        
        # Generate test states
        generator = HanoiDatasetGenerator(N, noise_rate=0.0)  # No noise for evaluation
        test_samples = generator.generate_dataset(num_samples=samples_per_N)
        
        if not test_samples:
            print(f"  No valid samples generated for N={N}")
            continue
        
        # Metrics
        teacher_accuracy = 0
        solve_success = 0
        avg_solve_length = 0
        avg_optimal_length = 0
        
        with torch.no_grad():
            for i, sample in enumerate(test_samples[:samples_per_N]):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(test_samples)}")
                
                state_tensor = torch.tensor(sample['state'], dtype=torch.long).unsqueeze(0).to(device)
                teacher_move = sample['teacher_move']
                
                # Test single-step accuracy (teacher move prediction)
                logits = model(state_tensor)
                probs = F.softmax(logits.view(-1), dim=0).view(3, 3)
                
                pred_move = torch.unravel_index(torch.argmax(logits.view(-1)), (3, 3))
                pred_move = (pred_move[0].item(), pred_move[1].item())
                
                if pred_move == tuple(teacher_move):
                    teacher_accuracy += 1
                
                # Test complete puzzle solving
                solve_result = solve_puzzle_with_model(
                    model, sample['state'], N, device, max_solve_steps
                )
                
                if solve_result['solved']:
                    solve_success += 1
                    avg_solve_length += solve_result['steps']
                
                # Get optimal solution length
                try:
                    initial_state = TowerOfHanoiState(N, np.array(sample['state']))
                    optimal_moves = solver.solve_analytical(initial_state, target_rod=2)
                    avg_optimal_length += len(optimal_moves)
                except:
                    avg_optimal_length += 2**N - 1  # Theoretical optimal for pure state
        
        # Compute averages
        num_samples = len(test_samples)
        teacher_acc = teacher_accuracy / num_samples
        solve_rate = solve_success / num_samples
        avg_solve_len = avg_solve_length / max(solve_success, 1)
        avg_opt_len = avg_optimal_length / num_samples
        
        results[N] = {
            'teacher_accuracy': teacher_acc,
            'solve_success_rate': solve_rate,
            'avg_solve_length': avg_solve_len,
            'avg_optimal_length': avg_opt_len,
            'efficiency_ratio': avg_opt_len / max(avg_solve_len, 1),
            'num_samples': num_samples
        }
        
        print(f"  Results for N={N}:")
        print(f"    Teacher Move Accuracy: {teacher_acc:.3f}")
        print(f"    Puzzle Solve Rate: {solve_rate:.3f}")
        print(f"    Avg Solve Length: {avg_solve_len:.1f}")
        print(f"    Avg Optimal Length: {avg_opt_len:.1f}")
        print(f"    Efficiency Ratio: {avg_opt_len/max(avg_solve_len,1):.3f}")
    
    return results'''


def solve_puzzle_with_model(model: HanoiNextStateTransformer, 
                           initial_state: np.ndarray, 
                           N: int,
                           device: str,
                           max_steps: int = 2000) -> Dict:
    """Solve a puzzle using the trained model."""
    model.eval()
    
    current_state = TowerOfHanoiState(N, initial_state.copy())
    steps = 0
    move_history = []
    
    with torch.no_grad():
        while steps < max_steps and not current_state.is_solved(target_rod=2):
            # Get model prediction
            state_tensor = torch.tensor(current_state.state, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(state_tensor)
            
            # Get legal moves
            legal_moves = current_state.get_legal_moves()
            if not legal_moves:
                break
            
            # Find best legal move according to model
            best_move = None
            best_prob = -1
            
            probs = F.softmax(logits.view(-1), dim=0).view(3, 3)
            
            for from_rod, to_rod in legal_moves:
                prob = probs[from_rod, to_rod].item()
                if prob > best_prob:
                    best_prob = prob
                    best_move = (from_rod, to_rod)
            
            if best_move is None:
                break
            
            # Make the move
            try:
                current_state = current_state.make_move(best_move[0], best_move[1])
                move_history.append(best_move)
                steps += 1
            except:
                break
    
    return {
        'solved': current_state.is_solved(target_rod=2),
        'steps': steps,
        'move_history': move_history,
        'final_state': current_state.state.copy()
    }

# Example training script with generalization testing
if __name__ == "__main__":
    # Configuration for training on N=3-10, testing on N=3-15
    config = {
        # Training Data - N=3 to 10
        'N_values': list(range(3, 11)),  # [3, 4, 5, 6, 7, 8, 9, 10]
        'samples_per_N': 3000,  # Fewer samples per N since we have more N values
        'noise_rate': 0.1,
        'val_split': 0.2,
        
        # Model - Support up to N=15
        'N_max': 15,  # Increased to support test cases
        'd_model': 256,  # Larger model for better generalization
        'n_heads': 8,
        'n_layers': 8,   # Deeper for better reasoning
        'd_ff': 1024,
        'dropout': 0.1,
        'K_fourier': 12,  # More Fourier features for size encoding
        'use_structural_bias': False,
        
        # Training
        'num_epochs': 150,  # More epochs for complex generalization
        'batch_size': 24,   # Smaller batch size due to variable lengths
        'learning_rate': 5e-4,  # Lower LR for stability
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'scheduler': 'cosine',
        'min_lr': 1e-6,
        
        # Loss
        'entropy_weight': 0.02,  # Slightly more exploration
        'label_smoothing': 0.05,  # Small amount of smoothing
        
        # Logging
        'project_name': 'hanoi-generalization-study',
        'run_name': f'N3-10_test15_structural',
        'save_every': 25,
        'checkpoint_dir': './checkpoints_generalization',
        'num_workers': 4
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Training on N={min(config['N_values'])}-{max(config['N_values'])}")
    print(f"Will test generalization up to N=15")
    
    # Create datasets
    # Train/val split
    """
    val_size = int(len(all_samples) * config['val_split'])
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    print(f"Dataset sizes - Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Create datasets
    train_dataset = HanoiDataset(train_samples)
    val_dataset = HanoiDataset(val_samples)
    
    # Create model
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = HanoiTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        use_wandb=False  # Set to True to enable wandb logging
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Test generalization
    print("\n" + "="*50)
    print("GENERALIZATION EVALUATION")
    print("="*50)
    
    # Test on training sizes (should be good)
    train_results = evaluate_generalization(
        model, 
        test_N_values=[3, 5, 8, 10],  # Sample from training range
        samples_per_N=200,
        device=device
    )
    
    # Test on unseen sizes (the real test!)
    generalization_results = evaluate_generalization(
        model,
        test_N_values=[11, 12, 13, 14, 15],  # Beyond training range
        samples_per_N=100,  # Fewer samples for larger N
        device=device
    )"""
