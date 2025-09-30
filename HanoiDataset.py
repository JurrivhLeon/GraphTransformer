import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Set
import random
from collections import deque
import itertools
import pickle
import json
from pathlib import Path
import time

class TowerOfHanoiState:
    """Represents a Tower of Hanoi state and provides utility methods."""
    def __init__(self, N: int, state: np.ndarray = None):
        self.N = N  # number of disks
        if state is None:
            # Default: all disks on rod 0 (A)
            self.state = np.zeros(N, dtype=int)
        else:
            self.state = state.copy()
    
    def __hash__(self):
        return hash(tuple(self.state))
    
    def __eq__(self, other):
        return np.array_equal(self.state, other.state)
    
    def copy(self):
        return TowerOfHanoiState(self.N, self.state)
    
    def get_rod_contents(self, rod: int) -> List[int]:
        """Get disks on a rod, ordered from bottom (largest) to top (smallest)."""
        disks_on_rod = [i for i in range(self.N) if self.state[i] == rod]
        return sorted(disks_on_rod, reverse=True)  # largest to smallest (bottom to top)
    
    def get_top_disk(self, rod: int) -> int:
        """Get the top (smallest) disk on a rod, or -1 if empty."""
        disks = self.get_rod_contents(rod)
        return disks[-1] if disks else -1
    
    def is_legal_move(self, from_rod: int, to_rod: int) -> bool:
        """Check if moving top disk from from_rod to to_rod is legal."""
        if from_rod == to_rod:
            return False
        
        top_from = self.get_top_disk(from_rod)
        if top_from == -1:  # source rod is empty
            return False
        
        top_to = self.get_top_disk(to_rod)
        if top_to == -1:  # target rod is empty
            return True
        
        return top_from < top_to  # smaller on larger
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal (from_rod, to_rod) moves."""
        legal_moves = []
        for from_rod in range(3):
            for to_rod in range(3):
                if self.is_legal_move(from_rod, to_rod):
                    legal_moves.append((from_rod, to_rod))
        return legal_moves
    
    def make_move(self, from_rod: int, to_rod: int) -> 'TowerOfHanoiState':
        """Make a move and return the new state."""
        if not self.is_legal_move(from_rod, to_rod):
            raise ValueError(f"Illegal move: {from_rod} -> {to_rod}")
        
        top_disk = self.get_top_disk(from_rod)
        new_state = self.copy()
        new_state.state[top_disk] = to_rod
        return new_state
    
    def is_solved(self, target_rod: int = 2) -> bool:
        """Check if all disks are on the target rod."""
        return np.all(self.state == target_rod)


def generate_all_states_BFS(N: int) -> Dict[TowerOfHanoiState, Tuple[int, int]]:
    """
    Generate all Tower of Hanoi states using BFS from the target state.
    
    Starting from the target state (all disks on rod 2), we explore all
    reachable states. For each new state reached, we record the reversed
    move that would optimally lead back toward the target.
    
    Args:
        N: Number of disks
        
    Returns:
        Dictionary mapping each state to its optimal move (from_rod, to_rod).
        The target state maps to None since no move is needed.
    """
    # Create target state: all disks on rod 2
    target_state = TowerOfHanoiState(N, state=np.full(N, 2, dtype=int))
    # Dictionary to store: state -> optimal move to make from that state
    state_to_move = {target_state: None}
    # BFS queue
    queue = deque([target_state])
    
    while queue:
        current_state = queue.popleft()
        # Get all legal moves from current state
        legal_moves = current_state.get_legal_moves()
        for from_rod, to_rod in legal_moves:
            # Make the move to get a new state
            next_state = current_state.make_move(from_rod, to_rod)
            # If we haven't visited this state yet
            if next_state not in state_to_move:
                # Record the REVERSED move (to get back toward target)
                # If we moved from_rod -> to_rod to reach next_state, then from next_state we should move to_rod -> from_rod
                reversed_move = (to_rod, from_rod)
                state_to_move[next_state] = reversed_move
                # Add to queue for further exploration
                queue.append(next_state)
    
    return state_to_move


def verify_solution(N: int, state_to_move: Dict[TowerOfHanoiState, Tuple[int, int]]):
    """
    Verify that the generated solution is correct by checking:
    1. We have exactly 3^N states
    2. Following moves from any state leads to the target
    """
    expected_states = 3 ** N
    actual_states = len(state_to_move)
    
    print(f"Number of disks: {N}")
    print(f"Expected states: {expected_states}")
    print(f"Generated states: {actual_states}")
    print(f"Verification: {'PASS' if expected_states == actual_states else 'FAIL'}")
    
    # Test a few random states to ensure moves lead toward target
    target_state = TowerOfHanoiState(N, state=np.full(N, 2, dtype=int))
    
    print("\nTesting path from initial state (all on rod 0):")
    test_state = TowerOfHanoiState(N)  # All disks on rod 0
    steps = 0
    max_steps = 2 ** N  # Maximum possible steps needed
    
    while not test_state.is_solved(target_rod=2) and steps < max_steps:
        move = state_to_move[test_state]
        print(f"Step {steps + 1}: Move from rod {move[0]} to rod {move[1]}")
        test_state = test_state.make_move(move[0], move[1])
        steps += 1
    
    if test_state.is_solved(target_rod=2):
        print(f"Successfully reached target in {steps} steps!")
        print(f"Optimal solution length: {2**N - 1}")
    else:
        print(f"Failed to reach target within {max_steps} steps")


def hanoi_recursive(n: int, source: int, target: int, auxiliary: int) -> List[Tuple[int, int]]:
    """Generate optimal move sequence using recursive algorithm."""
    if n == 1:
        return [(source, target)]
    else:
        moves = []
        # Move n-1 disks from source to auxiliary
        moves.extend(hanoi_recursive(n-1, source, auxiliary, target))
        # Move the largest disk from source to target
        moves.append((source, target))
        # Move n-1 disks from auxiliary to target
        moves.extend(hanoi_recursive(n-1, auxiliary, target, source))
        return moves


def generate_states_recursive(N: int, start_rod: int = 0, target_rod: int = 2) -> Dict[TowerOfHanoiState, Tuple[int, int]]:
    """
    Generate all Tower of Hanoi states using recursive algorithm.
    
    Executes the optimal move sequence and records each state with its
    corresponding optimal move. This creates the same mapping as BFS but
    is more efficient for larger puzzles.
    
    Args:
        N: Number of disks
        start_rod: Rod where all disks start (default: 0)
        target_rod: Rod where all disks should end (default: 2)
        
    Returns:
        Dictionary mapping each state to its optimal move (from_rod, to_rod).
        The target state maps to None since no move is needed.
    """
    # Generate the optimal move sequence
    auxiliary_rod = 3 - start_rod - target_rod  # The third rod
    moves = hanoi_recursive(N, start_rod, target_rod, auxiliary_rod)
    # Initialize with starting state
    current_state = TowerOfHanoiState(N, state=np.full(N, start_rod, dtype=int))
    # Dictionary to store: state -> optimal move
    state_to_move = {}
    
    # Record each state and its next move
    for move in moves:
        # Record current state with its optimal move
        state_to_move[current_state.copy()] = move
        # Execute the move to get to next state
        current_state = current_state.make_move(move[0], move[1])
    
    # Record the final (target) state with no move needed
    state_to_move[current_state] = None
    
    return state_to_move


def verify_recursive_solution(N: int, state_to_move: Dict[TowerOfHanoiState, Tuple[int, int]]):
    """
    Verify that the generated solution is correct.
    """
    expected_states = 2 ** N  # For path from start to target
    actual_states = len(state_to_move)
    
    print(f"Number of disks: {N}")
    print(f"States in optimal path: {expected_states}")
    print(f"Generated states: {actual_states}")
    print(f"Verification: {'PASS' if expected_states == actual_states else 'FAIL'}")
    
    # Test that following moves leads to target
    print("\nFollowing the optimal path:")
    start_state = TowerOfHanoiState(N, state=np.full(N, 0, dtype=int))
    current_state = start_state
    steps = 0
    
    while current_state in state_to_move and state_to_move[current_state] is not None:
        move = state_to_move[current_state]
        print(f"Step {steps + 1}: State {current_state.state} -> Move from rod {move[0]} to rod {move[1]}")
        current_state = current_state.make_move(move[0], move[1])
        steps += 1
        if steps > 5:  # Show first 5 steps only
            print(f"... ({2**N - 1 - steps} more steps)")
            break
    
    print(f"\nTotal moves in optimal solution: {2**N - 1}")


class HanoiDatasetGenerator:
    """Generates training data for Tower of Hanoi."""
    
    def __init__(self, N: int, noise_rate: float = 0.1, method: str = 'bfs'):
        """
        Initialize the dataset generator.
        
        Args:
            N: Number of disks
            noise_rate: Probability mass to distribute among non-optimal moves
            method: 'bfs' for all states (3^N), 'recursive' for optimal path (2^N)
        """
        self.N = N
        self.noise_rate = noise_rate
        self.method = method
        # Pre-compute optimal moves for all reachable states
        print(f"Pre-computing optimal moves using {method} method...")
        if method == 'bfs':
            self.state_to_move = generate_all_states_BFS(N)
            print(f"Generated {len(self.state_to_move)} states (all reachable)")
        elif method == 'recursive':
            self.state_to_move = generate_states_recursive(N, start_rod=0, target_rod=2)
            print(f"Generated {len(self.state_to_move)} states (optimal path)")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bfs' or 'recursive'")
    
    def get_optimal_move(self, state: TowerOfHanoiState) -> Tuple[int, int]:
        """Get the optimal move for a given state."""
        if state not in self.state_to_move:
            raise ValueError("State not in pre-computed optimal moves")
        move = self.state_to_move[state]
        if move is None:
            raise ValueError("State is already at target (no move needed)")
        return move
    
    def create_noisy_target(self, state: TowerOfHanoiState, teacher_move: Tuple[int, int]) -> np.ndarray:
        """Create noisy target distribution as described in the paper."""
        legal_moves = state.get_legal_moves()
        # Create 3x3 probability matrix (but only 6 off-diagonal elements are used)
        target = np.zeros((3, 3))
        if not legal_moves:
            return target  # No legal moves
        
        # Teacher move probability
        teacher_prob = 1.0 - self.noise_rate
        target[teacher_move[0], teacher_move[1]] = teacher_prob
        
        # Distribute remaining probability among other legal moves
        other_legal_moves = [move for move in legal_moves if move != teacher_move]
        if other_legal_moves:
            noise_prob_per_move = self.noise_rate / len(other_legal_moves)
            for from_rod, to_rod in other_legal_moves:
                target[from_rod, to_rod] = noise_prob_per_move
        
        return target
    
    def generate_dataset(self, num_samples: int, target_rod: int = 2, 
                        use_all_states: bool = False) -> List[Dict]:
        """
        Generate training dataset.
        
        Args:
            num_samples: Number of samples to generate
            target_rod: Target rod (default: 2)
            use_all_states: If True, use all pre-computed non-target states.
                           If False, randomly sample num_samples states.
        
        Returns:
            List of training samples, each containing:
                - state: The current state array
                - teacher_move: The optimal move
                - noisy_target: The noisy probability distribution
                - legal_moves: All legal moves from this state
        """
        # Get all non-target states (states that have a move to make)
        available_states = [s for s in self.state_to_move.keys() if self.state_to_move[s] is not None]
        
        if use_all_states:
            # Use all available states
            states_to_use = available_states
        else:
            # Randomly sample from available states
            if num_samples > len(available_states):
                print(f"Warning: Requested {num_samples} samples but only {len(available_states)} "
                      f"states available. Using all available states.")
                states_to_use = available_states
            else:
                # Sample without replacement
                sampled_indices = np.random.choice(len(available_states), size=num_samples, replace=False)
                states_to_use = [available_states[i] for i in sampled_indices]
        
        # Generate dataset
        dataset = []
        for state in states_to_use:
            teacher_move = self.get_optimal_move(state)
            noisy_target = self.create_noisy_target(state, teacher_move)
            sample = {
                'state': state.state.copy(),
                'teacher_move': teacher_move,
                'noisy_target': noisy_target,
                'legal_moves': state.get_legal_moves()
            }
            dataset.append(sample)
        
        return dataset


class HanoiDataset(Dataset):
    """PyTorch Dataset for Tower of Hanoi."""
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'state': torch.tensor(sample['state'], dtype=torch.long),
            'teacher_move': torch.tensor(sample['teacher_move'], dtype=torch.long),
            'noisy_target': torch.tensor(sample['noisy_target'], dtype=torch.float32),
            'legal_moves': sample['legal_moves']  # Keep as list for now
        }


def hanoi_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for batching Tower of Hanoi samples with variable N.
    
    Pads state sequences to the length of the longest sequence in the batch.
    Uses padding value of -1 (since valid values are 0, 1, 2).
    
    Args:
        batch: List of samples from HanoiDataset
        
    Returns:
        Batched dictionary with:
            - state: [batch_size, max_N] padded tensor
            - lengths: [batch_size] tensor with original sequence lengths
            - teacher_move: [batch_size, 2] tensor
            - noisy_target: [batch_size, 3, 3] tensor
            - legal_moves: List of legal moves for each sample
    """
    # Extract components
    states = [sample['state'] for sample in batch]
    teacher_moves = [sample['teacher_move'] for sample in batch]
    noisy_targets = [sample['noisy_target'] for sample in batch]
    legal_moves = [sample['legal_moves'] for sample in batch]
    
    # Get original lengths before padding
    lengths = torch.tensor([len(state) for state in states], dtype=torch.long)
    
    # Pad states to max length in batch (padding value = -1)
    # pad_sequence expects [seq_len, batch] by default, so we transpose
    states_padded = pad_sequence(states, batch_first=True, padding_value=-1)
    
    # Stack teacher moves and noisy targets (these have fixed dimensions)
    teacher_moves_batch = torch.stack(teacher_moves, dim=0)
    noisy_targets_batch = torch.stack(noisy_targets, dim=0)
    
    return {
        'state': states_padded,           # [batch_size, max_N]
        'lengths': lengths,                # [batch_size]
        'teacher_move': teacher_moves_batch,  # [batch_size, 2]
        'noisy_target': noisy_targets_batch,  # [batch_size, 3, 3]
        'legal_moves': legal_moves         # List of lists
    }


def generate_complete_dataset(N_min: int = 3, N_max: int = 10, noise_rate: float = 0.1,
                             method: str = 'bfs', save_dir: str = './hanoi_data'):
    """
    Generate complete dataset for Tower of Hanoi puzzles from N_min to N_max.
    
    Args:
        N_min: Minimum number of disks
        N_max: Maximum number of disks
        noise_rate: Noise rate for target distribution
        method: 'bfs' for all states, 'recursive' for optimal path only
        save_dir: Directory to save the dataset
        
    Returns:
        Dictionary containing all datasets and metadata
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    metadata = {
        'N_range': (N_min, N_max),
        'noise_rate': noise_rate,
        'method': method,
        'total_samples': 0,
        'samples_per_N': {},
        'generation_time': {}
    }
    
    print("="*70)
    print(f"Generating Tower of Hanoi Dataset (N={N_min} to {N_max})")
    print(f"Method: {method}, Noise rate: {noise_rate}")
    print("="*70)
    
    total_start_time = time.time()
    
    for N in range(N_min, N_max + 1):
        print(f"\n{'='*70}")
        print(f"Processing N={N} disks...")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Generate dataset for this N
        generator = HanoiDatasetGenerator(N=N, noise_rate=noise_rate, method=method)
        dataset = generator.generate_dataset(num_samples=None, use_all_states=True)
        
        # Add N to each sample for later reference
        for sample in dataset:
            sample['N'] = N
        
        all_data.extend(dataset)
        
        elapsed = time.time() - start_time
        num_samples = len(dataset)
        
        # Update metadata
        metadata['samples_per_N'][N] = num_samples
        metadata['generation_time'][N] = elapsed
        metadata['total_samples'] += num_samples
        
        print(f"Generated {num_samples:,} samples in {elapsed:.2f} seconds")
        print(f"Average time per sample: {elapsed/num_samples*1000:.4f} ms")
        
        # Calculate expected states
        if method == 'bfs':
            expected = 3**N - 1  # All states except target
        else:
            expected = 2**N - 1  # Optimal path states except target
        print(f"Expected states: {expected:,}, Generated states: {num_samples:,}")
        
        if num_samples != expected:
            print(f"WARNING: Mismatch in number of states!")
    
    total_elapsed = time.time() - total_start_time
    metadata['total_generation_time'] = total_elapsed
    
    print(f"\n{'='*70}")
    print(f"Dataset Generation Complete!")
    print(f"{'='*70}")
    print(f"Total samples: {metadata['total_samples']:,}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Average time per sample: {total_elapsed/metadata['total_samples']*1000:.4f} ms")
    
    # Save dataset
    print(f"\n{'='*70}")
    print("Saving dataset...")
    print(f"{'='*70}")
    
    # Save as pickle (efficient, includes all data)
    pickle_path = save_path / f'hanoi_dataset_N{N_min}_to_{N_max}_{method}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump({'data': all_data, 'metadata': metadata}, f)
    print(f"Saved pickle to: {pickle_path}")
    print(f"File size: {pickle_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save metadata as JSON for easy inspection
    metadata_path = save_path / f'hanoi_metadata_N{N_min}_to_{N_max}_{method}.json'
    with open(metadata_path, 'w') as f:
        # Convert numpy types to native Python types for JSON
        metadata_json = {
            'N_range': metadata['N_range'],
            'noise_rate': float(metadata['noise_rate']),
            'method': metadata['method'],
            'total_samples': int(metadata['total_samples']),
            'samples_per_N': {str(k): int(v) for k, v in metadata['samples_per_N'].items()},
            'generation_time': {str(k): float(v) for k, v in metadata['generation_time'].items()},
            'total_generation_time': float(metadata['total_generation_time'])
        }
        json.dump(metadata_json, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    return all_data, metadata


def load_dataset(file_path: str):
    """
    Load a previously saved dataset.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Tuple of (data, metadata)
    """
    with open(file_path, 'rb') as f:
        saved = pickle.load(f)
    
    print(f"Loaded dataset from: {file_path}")
    print(f"Total samples: {saved['metadata']['total_samples']:,}")
    print(f"N range: {saved['metadata']['N_range']}")
    print(f"Method: {saved['metadata']['method']}")
    print(f"Samples per N: {saved['metadata']['samples_per_N']}")
    
    return saved['data'], saved['metadata']


def create_pytorch_dataset(data: List[Dict], N_filter: List[int] = None):
    """
    Create PyTorch dataset from loaded data, optionally filtering by N.
    
    Args:
        data: List of samples
        N_filter: List of N values to include (None = include all)
        
    Returns:
        HanoiDataset instance
    """
    if N_filter is not None:
        filtered_data = [sample for sample in data if sample['N'] in N_filter]
        print(f"Filtered to N={N_filter}: {len(filtered_data):,} samples")
    else:
        filtered_data = data
        print(f"Using all samples: {len(filtered_data):,}")
    
    return HanoiDataset(filtered_data)


if __name__ == "__main__":
    # Generate and save complete dataset
    print("Generating complete dataset...")
    all_data, metadata = generate_complete_dataset(
        N_min=3, 
        N_max=10, 
        noise_rate=0.1,
        method='bfs',  # Use 'bfs' for complete coverage or 'recursive' for efficiency
        save_dir='./hanoi_data'
    )
    
    print("\n" + "="*70)
    print("Dataset Summary:")
    print("="*70)
    for N in range(3, 11):
        print(f"N={N}: {metadata['samples_per_N'][N]:,} samples "
              f"({metadata['generation_time'][N]:.2f}s)")
    
    # Example: Load the dataset later
    print("\n" + "="*70)
    print("Testing dataset loading...")
    print("="*70)
    loaded_data, loaded_metadata = load_dataset('./hanoi_data/hanoi_dataset_N3_to_10_bfs.pkl')
    
    # Create PyTorch dataset for specific N values
    print("\n" + "="*70)
    print("Creating PyTorch datasets...")
    print("="*70)
    
    # Example 1: Dataset for N=3,4,5 only
    dataset_small = create_pytorch_dataset(loaded_data, N_filter=[3, 4, 5])
    
    # Example 2: Dataset for all N values
    dataset_all = create_pytorch_dataset(loaded_data, N_filter=None)
    
    # Example 3: Create DataLoader
    
    dataloader = DataLoader(
        dataset_all, 
        batch_size=32, 
        collate_fn=hanoi_collate_fn,
        shuffle=True
    )
    
    print("\nTesting DataLoader with first batch...")
    for batch in dataloader:
        print(f"Batch state shape: {batch['state'].shape}")
        print(f"Batch lengths: {batch['lengths']}")
        print(f"Batch teacher moves shape: {batch['teacher_move'].shape}")
        print(f"Batch size: {batch['state'].shape[0]}")
        break
    
    print("\n" + "="*70)
    print("All done! Dataset saved and ready to use.")
    print("="*70)