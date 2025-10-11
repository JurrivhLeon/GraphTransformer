import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Union
from HanoiDataset import load_dataset, create_pytorch_dataset, hanoi_next_state_collate_fn

class FourierFeatures(nn.Module):
    """Generates sinusoidal (Fourier) features for size encoding with optional caching."""
    
    def __init__(self, K: int = 8, beta: float = 10000.0, precompute_limit: int = 64):
        super().__init__()
        self.K = K
        self.beta = beta
        self.precompute_limit = precompute_limit
        
        # Pre-compute frequency weights
        k_values = torch.arange(K, dtype=torch.float32)
        omega = beta ** (-2 * k_values / (2 * K))
        self.register_buffer('omega', omega)

        # Optional cached features for indices in [0, precompute_limit]
        if precompute_limit is not None and precompute_limit >= 0:
            with torch.no_grad():
                cached_indices = torch.arange(precompute_limit, dtype=torch.float32)
                omega_i = cached_indices.unsqueeze(-1) * self.omega  # (L, K)
                sin_features = torch.sin(omega_i)
                cos_features = torch.cos(omega_i)
                cached = torch.stack([sin_features, cos_features], dim=-1).view(precompute_limit, -1)  # (L, 2*K)
            self.register_buffer('precomputed_features', cached)
        else:
            self.precomputed_features = None
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: tensor of shape (...,) with integer-like indices (0 <= idx <= precompute_limit)
        Returns:
            features: tensor of shape (..., 2*K) with sin/cos features
        """
        if getattr(self, 'precomputed_features', None) is None:
            raise RuntimeError("FourierFeatures: precomputed_features buffer is missing.")
        idx_long = indices.long()
        if (idx_long.min() < 0) or (idx_long.max() > self.precompute_limit):
            raise ValueError(
                f"FourierFeatures indices out of range: got [{idx_long.min().item()}, {idx_long.max().item()}], "
                f"allowed 0..{self.precompute_limit}"
            )
        return self.precomputed_features[idx_long].to(indices.device)  # (B, L, 2*K)


class StateEncoder(nn.Module):
    """Encodes Tower of Hanoi state into node embeddings with padding support.
    Rod nodes are placed first (positions 0..2), followed by disk nodes (positions 3..3+N_max-1).
    """
    
    def __init__(self, N_max: int, d_model: int, K_fourier: int = 8, rod_id_dim: int = 8):
        super().__init__()
        self.N_max = N_max
        self.d_model = d_model
        self.K_fourier = K_fourier
        self.rod_id_dim = rod_id_dim
        
        # Fourier features for disk sizes
        self.fourier_features = FourierFeatures(K_fourier)
        
        # Type embeddings
        self.register_buffer('disk_type', torch.tensor([1.0, 0.0]))
        self.register_buffer('rod_type', torch.tensor([0.0, 1.0]))
        
        # Rod embeddings (one-hot)
        rod_embeddings = torch.eye(3)  # 3x3 identity
        self.register_buffer('rod_embeddings', rod_embeddings)
        # Learnable rod ID embedding
        self.rod_id_embed = nn.Embedding(3, rod_id_dim)
        
        # Projection layers
        disk_feature_dim = 2 + 2*K_fourier + 3 + rod_id_dim  # type + fourier + position_info + rod_id_embed
        rod_feature_dim = 2 + 1 + 1 + 2*2*K_fourier + rod_id_dim  # add rod id embedding
        
        self.disk_projection = nn.Linear(disk_feature_dim, d_model)
        self.rod_projection = nn.Linear(rod_feature_dim, d_model)
        
        # Learnable NULL vectors for empty rods
        self.null_fourier = nn.Parameter(torch.zeros(2*K_fourier))
        
        # Padding token embedding (for padded disk positions)
        self.padding_embedding = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, state: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            state: (batch_size, max_N) where state[b,i] = rod of disk i, or -1 for padding
            lengths: (batch_size,) actual number of disks for each sample (optional)
        Returns:
            embeddings: (batch_size, max_N+3, d_model)
            state_info: dictionary with computed state information
        """
        batch_size, max_N = state.shape
        device = state.device
        
        # Infer lengths if not provided
        if lengths is None:
            # Count non-padding tokens (-1 is padding)
            lengths = (state != -1).sum(dim=1)  # (batch_size,)
        
        # Create padding mask: True for valid positions, False for padding
        padding_mask = state != -1  # (batch_size, max_N)
        
        # Compute state information (handles padding internally)
        state_info = self._compute_state_info(state, lengths, padding_mask)
        
        embeddings = []

        # === ROD NODE EMBEDDINGS (first) ===
        for k in range(3):
            # Type embedding
            rod_type = self.rod_type.unsqueeze(0).expand(batch_size, -1)  # (B, 2)
            # Length and emptiness information
            len_norm = state_info['len_norm'][:, k:k+1]  # (B, 1)
            is_empty = state_info['is_empty'][:, k:k+1].float()  # (B, 1)
            # Top disk Fourier features (NULL if empty)
            top_disk_indices = state_info['top_disk'][:, k].float()
            empty_mask = state_info['is_empty'][:, k]
            top_indices_safe = torch.clamp(top_disk_indices, min=0)
            top_fourier = self.fourier_features(top_indices_safe)
            top_fourier = torch.where(empty_mask.unsqueeze(1), self.null_fourier.unsqueeze(0).expand(batch_size, -1).to(state.device), top_fourier)
            # Bottom disk Fourier features (NULL if empty)
            bottom_disk_indices = state_info['bottom_disk'][:, k].float()
            bottom_indices_safe = torch.clamp(bottom_disk_indices, min=0)
            bottom_fourier = self.fourier_features(bottom_indices_safe)
            bottom_fourier = torch.where(empty_mask.unsqueeze(1), self.null_fourier.unsqueeze(0).expand(batch_size, -1).to(state.device), bottom_fourier)
            # Rod id embedding
            rod_ids = torch.full((batch_size,), k, dtype=torch.long, device=state.device)
            rod_id_vec = self.rod_id_embed(rod_ids)
            # Concatenate and project
            rod_features = torch.cat([rod_type, len_norm, is_empty, top_fourier, bottom_fourier, rod_id_vec], dim=1)
            rod_embedding = self.rod_projection(rod_features)
            embeddings.append(rod_embedding)

        # === DISK NODE EMBEDDINGS (after rods) ===
        for i in range(max_N):
            # Check which samples have this disk (not padded)
            valid_mask = padding_mask[:, i]  # (batch_size,)
            
            if not valid_mask.any():
                # All samples have padding at this position - use padding embedding
                disk_embedding = self.padding_embedding.unsqueeze(0).expand(batch_size, -1)
                embeddings.append(disk_embedding)
                continue
            
            # Type embedding
            disk_type = self.disk_type.unsqueeze(0).expand(batch_size, -1)  # (B, 2)
            
            # Fourier size features
            disk_indices = torch.full((batch_size,), i, device=device, dtype=torch.float32)
            fourier_size = self.fourier_features(disk_indices)  # (B, 2*K)
            
            # Position information within stack
            depth_norm = state_info['depth_norm'][:, i:i+1]  # (B, 1)
            is_top = state_info['is_top'][:, i:i+1].float()  # (B, 1)
            is_bottom = state_info['is_bottom'][:, i:i+1].float()  # (B, 1)
            position_info = torch.cat([depth_norm, is_top, is_bottom], dim=1)  # (B, 3)
            
            # Rod information (which rod this disk is on) via learned embedding
            rod_indices = state[:, i].clamp(min=0).long()  # Clamp -1 to 0 temporarily
            rod_id_vec = self.rod_id_embed(rod_indices)  # (B, 8)
            
            # Concatenate all disk features
            disk_features = torch.cat([disk_type, fourier_size, position_info, rod_id_vec], dim=1)
            disk_embedding = self.disk_projection(disk_features)  # (B, d_model)
            
            # Replace embeddings for padded positions with padding embedding
            disk_embedding = torch.where(
                valid_mask.unsqueeze(1),
                disk_embedding,
                self.padding_embedding.unsqueeze(0).expand(batch_size, -1)
            )
            
            embeddings.append(disk_embedding)
        
        # Stack all embeddings
        embeddings = torch.stack(embeddings, dim=1)  # (B, 3+max_N, d_model) rods-first
        
        # Store mask for attention
        state_info['padding_mask'] = padding_mask
        state_info['lengths'] = lengths
        
        return embeddings, state_info
    
    def _compute_state_info(self, state: torch.Tensor, lengths: torch.Tensor, padding_mask: torch.Tensor) -> Dict:
        """Compute auxiliary state information with padding support."""
        batch_size, max_N = state.shape
        device = state.device
        
        # Initialize outputs
        depth_norm = torch.zeros(batch_size, max_N, device=device)
        is_top = torch.zeros(batch_size, max_N, dtype=torch.bool, device=device)
        is_bottom = torch.zeros(batch_size, max_N, dtype=torch.bool, device=device)
        
        len_norm = torch.zeros(batch_size, 3, device=device)
        is_empty = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
        top_disk = torch.full((batch_size, 3), -1, dtype=torch.long, device=device)
        bottom_disk = torch.full((batch_size, 3), -1, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            N = lengths[b].item()
            
            for k in range(3):  # For each rod
                # Find disks on this rod (only consider valid, non-padded disks)
                disks_on_rod = []
                for i in range(N):  # Only check up to actual length
                    if state[b, i] == k:
                        disks_on_rod.append(i)
                
                if not disks_on_rod:
                    is_empty[b, k] = True
                    len_norm[b, k] = 0.0
                    continue
                
                # Sort disks on rod (largest to smallest, bottom to top)
                disks_on_rod.sort(reverse=True)
                
                len_norm[b, k] = len(disks_on_rod) / max(N, 1)
                top_disk[b, k] = disks_on_rod[-1]  # Smallest disk (top)
                bottom_disk[b, k] = disks_on_rod[0]  # Largest disk (bottom)
                
                # Compute per-disk information
                height_k = len(disks_on_rod)
                for idx, disk in enumerate(disks_on_rod):
                    above_i = height_k - 1 - idx  # Number of disks above
                    is_top[b, disk] = (above_i == 0)
                    is_bottom[b, disk] = (idx == 0)
                    
                    if height_k > 1:
                        depth_norm[b, disk] = above_i / (height_k - 1)
                    else:
                        depth_norm[b, disk] = 0.0
        
        return {
            'depth_norm': depth_norm,
            'is_top': is_top, 
            'is_bottom': is_bottom,
            'len_norm': len_norm,
            'is_empty': is_empty,
            'top_disk': top_disk,
            'bottom_disk': bottom_disk
        }


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional structural biases for graph transformer."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_structural_bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_structural_bias = use_structural_bias
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False) 
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Structural bias parameters (learnable) - only if enabled
        if self.use_structural_bias:
            self.edge_type_bias = nn.Parameter(torch.zeros(4))  # disk->disk, disk->rod, rod->disk, rod->rod
            self.same_rod_bias = nn.Parameter(torch.zeros(2))   # same rod vs different rods
            self.chain_distance_bias = nn.Parameter(torch.zeros(4))  # distances 0,1,2,3+
            self.relative_size_bias = nn.Parameter(torch.zeros(3))   # -1, 0, +1
    
    def forward(self, x: torch.Tensor, N: int, state_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, max_N+3, d_model)
            N: maximum number of disks in batch
            state_info: auxiliary state information (required if use_structural_bias=True)
                       should contain 'padding_mask' for masking attention
        Returns:
            output: (batch_size, max_N+3, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Pre-layer norm
        x_norm = F.layer_norm(x, (d_model,))
        
        # Compute Q, K, V
        Q = self.W_Q(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, seq_len, seq_len)
        
        # Add structural biases if enabled
        if self.use_structural_bias and state_info is not None:
            bias = self._compute_structural_bias(batch_size, N, state_info)  # (B, seq_len, seq_len)
            scores = scores + bias.unsqueeze(1)  # Broadcast over heads
        
        # Apply padding mask to attention scores
        if state_info is not None and 'padding_mask' in state_info:
            padding_mask = state_info['padding_mask']  # (B, max_N)
            # Rods are first and never padded; disks follow
            rod_mask = torch.ones(batch_size, 3, dtype=torch.bool, device=padding_mask.device)
            full_mask = torch.cat([rod_mask, padding_mask], dim=1)  # (B, 3+max_N)
            
            # Create attention mask: (B, 1, 1, seq_len) for broadcasting
            # Mask out attention TO padded positions
            attn_mask = full_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len)
            scores = scores.masked_fill(~attn_mask, float('-inf'))
            
            # Also mask out attention FROM padded positions (optional, but cleaner)
            attn_mask_from = full_mask.unsqueeze(1).unsqueeze(3)  # (B, 1, seq_len, 1)
            scores = scores.masked_fill(~attn_mask_from, float('-inf'))
        
        # Apply attention
        attn = F.softmax(scores, dim=-1)
        # Set NaN values (from -inf softmax) to 0
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, V)  # (B, H, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        out = self.W_O(out)
        
        return x + out  # Residual connection
    
    def _compute_structural_bias(self, batch_size: int, N: int, state_info: Dict) -> torch.Tensor:
        """Compute structural attention biases."""
        seq_len = N + 3
        device = next(iter(state_info.values())).device
        bias = torch.zeros(batch_size, seq_len, seq_len, device=device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if i == j:
                        continue
                    
                    # Determine node types with rods-first layout
                    is_i_disk = i >= 3
                    is_j_disk = j >= 3
                    
                    # Edge type bias
                    if is_i_disk and is_j_disk:
                        bias[b, i, j] += self.edge_type_bias[0]  # disk->disk
                    elif is_i_disk and not is_j_disk:
                        bias[b, i, j] += self.edge_type_bias[1]  # disk->rod
                    elif not is_i_disk and is_j_disk:
                        bias[b, i, j] += self.edge_type_bias[2]  # rod->disk
                    else:
                        bias[b, i, j] += self.edge_type_bias[3]  # rod->rod
                    
                    # Same rod bias (for disk-disk and disk-rod pairs)
                    if is_i_disk and is_j_disk:
                        # Disks are at indices offset by 3 in embeddings, but state indices are 0..N-1
                        # Map embedding index to disk index by (i-3)
                        rod_i = state_info['state'][b, i - 3]
                        rod_j = state_info['state'][b, j - 3]
                        if rod_i == rod_j:
                            bias[b, i, j] += self.same_rod_bias[0]
                        else:
                            bias[b, i, j] += self.same_rod_bias[1]
                    
                    # Relative size bias (for disk-disk pairs)
                    if is_i_disk and is_j_disk:
                        size_diff = i - j  # Since disks are ordered by size
                        if size_diff < 0:
                            bias[b, i, j] += self.relative_size_bias[0]  # i smaller than j
                        elif size_diff == 0:
                            bias[b, i, j] += self.relative_size_bias[1]  # same size (shouldn't happen)
                        else:
                            bias[b, i, j] += self.relative_size_bias[2]  # i larger than j
        
        return bias


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-layer norm."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_structural_bias: bool = True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, use_structural_bias)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, N: int, state_info: Optional[Dict] = None) -> torch.Tensor:
        # Pre-norm attention with residual
        attn_out = self.attention(x, N, state_info)
        
        # Pre-norm feed-forward with residual
        ff_input = attn_out
        ff_out = ff_input + self.feed_forward(self.norm2(ff_input))
        
        return ff_out


class ReadoutHead(nn.Module):
    """Readout head that predicts next state by outputting rod assignments for each disk."""
    
    def __init__(self, d_model: int, N_max: int):
        super().__init__()
        self.N_max = N_max
        # MLP to predict rod assignment for each disk
        self.disk_rod_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)  # 3 rods
        )
    
    def forward(self, embeddings: torch.Tensor, N: int, state_info: Dict) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, 3+N_max, d_model) with rods first, then disks
            N: number of disks in batch
            state_info: state information for masking
        Returns:
            logits: (batch_size, N_max, 3) logits for each disk's rod assignment
        """
        batch_size = embeddings.shape[0]
        
        # Extract disk embeddings (positions 3:3+N_max)
        disk_embeddings = embeddings[:, 3:3+self.N_max, :]  # (B, N_max, d_model)
        
        # Predict rod assignment for each disk
        logits = self.disk_rod_predictor(disk_embeddings)  # (B, N_max, 3)
        
        # Apply masking for padded positions
        if 'padding_mask' in state_info:
            padding_mask = state_info['padding_mask']  # (B, N_max)
            # Mask out padded positions with large negative values
            logits = logits.masked_fill(~padding_mask.unsqueeze(-1), -1e9)
        
        return logits


class HanoiNextStateTransformer(nn.Module):
    """Complete Graph Transformer for Tower of Hanoi with padding support."""

    def __init__(self, N_max: int, d_model: int = 128, n_heads: int = 8, rod_id_dim: int = 8,
                 n_layers: int = 6, d_ff: int = 512, dropout: float = 0.1, 
                 K_fourier: int = 8, use_structural_bias: bool = True):
        super().__init__()
        self.N_max = N_max
        self.d_model = d_model
        self.use_structural_bias = use_structural_bias
        
        # Components
        self.encoder = StateEncoder(N_max, d_model, K_fourier, rod_id_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_structural_bias) 
            for _ in range(n_layers)
        ])
        self.readout = ReadoutHead(d_model, N_max)
    
    def forward(self, batch: Union[torch.Tensor, Dict]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            batch: Either a tensor (batch_size, max_N) or a dictionary containing:
                - 'state': (batch_size, max_N) where state[b,i] = rod of disk i, or -1 for padding
                - 'lengths': (batch_size,) actual number of disks for each sample (optional)
                - 'legal_moves': List of legal moves for each sample (optional)
        Returns:
            logits: (batch_size, N_max, 3) logits for each disk's rod assignment
            state_info: Dictionary with computed state information
        """
        # Get device dynamically
        device = next(self.parameters()).device
        
        # Handle both tensor and dictionary inputs
        if isinstance(batch, torch.Tensor):
            state = batch.to(device)
            lengths = None
            legal_moves = None
        else:
            state = batch['state'].to(device)
            lengths = batch.get('lengths')
            if lengths is not None:
                lengths = lengths.to(device)
            legal_moves = batch.get('legal_moves')
        
        batch_size, max_N = state.shape
        
        # Encode state (handles padding internally)
        embeddings, state_info = self.encoder(state, lengths)
        state_info['legal_moves'] = legal_moves
        
        # Add original state for bias computation (mask out padding with valid values)
        state_masked = state.clone()
        if lengths is not None:
            for b in range(batch_size):
                N = lengths[b].item()
                if N < max_N:
                    state_masked[b, N:] = 0  # Replace padding with valid rod index
        else:
            # Infer lengths and mask
            lengths = (state != -1).sum(dim=1)
            for b in range(batch_size):
                N = lengths[b].item()
                if N < max_N:
                    state_masked[b, N:] = 0
        
        state_info['state'] = state_masked
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            embeddings = block(embeddings, max_N, state_info if self.use_structural_bias else state_info)
        
        # Readout (only uses rod embeddings, unaffected by padding)
        logits = self.readout(embeddings, max_N, state_info)
        next_moves = self.extract_next_move(logits, state_info)
        
        return logits, next_moves, state_info
    
    def extract_next_move(self, logits: torch.Tensor, state_info: Dict) -> torch.Tensor:
        """
        Extract the next move by finding legal moves and choosing the one with highest probability.
        Args:
            logits: (batch_size, N_max, 3) logits for each disk's rod assignment
            state_info: Dictionary with computed state information
        Returns:
            next_moves: (batch_size, 2) predicted next moves
        """
        batch_size = logits.shape[0]

        probs = F.softmax(logits, dim=-1)  # (B, N_max, 3)
        
        # Get legal moves for each batch
        if 'legal_moves' in state_info and state_info['legal_moves'] is not None:
            legal_moves_list = state_info['legal_moves']
        else:
            legal_moves_list = [self._get_legal_moves(state_info, batch_idx) for batch_idx in range(batch_size)]
        
        # Initialize next moves tensor
        device = next(self.parameters()).device
        next_moves = torch.zeros(batch_size, 2, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            legal_moves = legal_moves_list[b]
            if not legal_moves:
                # No legal moves, keep default (0, 0)
                continue
            
            # For each legal move, compute the probability of the resulting state
            move_probs = []
            for from_rod, to_rod in legal_moves:
                # Find the disk being moved (top disk on from_rod)
                top_disk = state_info['top_disk'][b, from_rod].item()
                if top_disk == -1:  # No disk on this rod
                    continue
                # Get probability of this disk being on the target rod
                disk_prob = probs[b, top_disk, to_rod].item()
                move_probs.append((disk_prob, from_rod, to_rod))
            if not move_probs:
                continue
            
            # Choose the move with highest probability
            _, best_from, best_to = max(move_probs, key=lambda x: x[0])
            next_moves[b, 0] = best_from
            next_moves[b, 1] = best_to
        
        return next_moves
        
    
    def _get_legal_moves(self, state_info: Dict, batch_idx: int) -> List[Tuple[int, int]]:
        """Get all legal moves for a given batch index."""
        legal_moves = []
        
        for from_rod in range(3):
            for to_rod in range(3):
                if from_rod != to_rod and self._is_move_legal(state_info, batch_idx, from_rod, to_rod):
                    legal_moves.append((from_rod, to_rod))
        
        return legal_moves
    
    def _is_move_legal(self, state_info: Dict, batch_idx: int, from_rod: int, to_rod: int) -> bool:
        """Check if a move is legal."""
        # Source rod must not be empty
        if state_info['is_empty'][batch_idx, from_rod]:
            return False
        
        # If target rod is empty, move is legal
        if state_info['is_empty'][batch_idx, to_rod]:
            return True
        
        # Otherwise, check size constraint
        top_from = state_info['top_disk'][batch_idx, from_rod].item()
        top_to = state_info['top_disk'][batch_idx, to_rod].item()
        
        return top_from < top_to  # Smaller disk on larger disk


# Example usage
if __name__ == "__main__":
    print("=== Testing Model ===")
    loaded_data, loaded_metadata = load_dataset('./hanoi_data/hanoi_dataset_N3_to_10_bfs.pkl')
    
    # Create PyTorch dataset for specific N values
    print("\n" + "="*70)
    print("Creating PyTorch datasets...")
    print("="*70)
    
    dataset_small = create_pytorch_dataset(loaded_data, N_filter=[3, 4, 5, 6])
    dataloader = DataLoader(
        dataset_small, 
        batch_size=32, 
        collate_fn=hanoi_next_state_collate_fn,
        shuffle=True
    )

    # Test with same-length batch (traditional)
    N = 4
    batch_size = 32
    model = HanoiNextStateTransformer(N_max=10, d_model=64, n_heads=4, n_layers=3, use_structural_bias=False)

    for batch in dataloader:
        print(f"Batch states shape: {batch['state'].shape}")
        print(f"Batch targets shape: {batch['noisy_target'].shape}")
        logits, next_moves, state_info = model(batch)
        print(f"Output logits shape: {logits.shape}")
        print(f"Predicted next moves shape: {next_moves.shape}")
        break  # Just show one batch