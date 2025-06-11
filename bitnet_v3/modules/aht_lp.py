"""
Adaptive Hadamard Transform with Learnable Parameters (AHT-LP) for BitNet v3.

Implements the enhanced Hadamard transformation with learnable scaling and shifting parameters:
H_adaptive(x) = γ ⊙ (H_m · x) + β

where γ and β are learnable parameters that adapt to activation distributions,
and H_m is the Hadamard matrix.

The parameters are updated using separate learning rates:
γ_{t+1} = γ_t - η_γ ∇_γ L
β_{t+1} = β_t - η_β ∇_β L
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import warnings

from ..core.hadamard import (
    hadamard_transform, 
    fast_hadamard_transform,
    create_hadamard_matrix,
    is_power_of_2,
    next_power_of_2
)


class LearnableHadamardParameters(nn.Module):
    """
    Learnable scale and shift parameters for adaptive Hadamard transform.
    """
    
    def __init__(
        self,
        size: int,
        init_scale: float = 1.0,
        init_shift: float = 0.0,
        scale_lr_multiplier: float = 1.0,
        shift_lr_multiplier: float = 1.0,
        scale_weight_decay: float = 0.0,
        shift_weight_decay: float = 0.0,
        constraint_scale: Optional[Tuple[float, float]] = None,
        constraint_shift: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        
        self.size = size
        self.constraint_scale = constraint_scale
        self.constraint_shift = constraint_shift
        
        # Learnable parameters
        self.scale = nn.Parameter(torch.full((size,), init_scale))
        self.shift = nn.Parameter(torch.full((size,), init_shift))
        
        # Set learning rate multipliers as attributes
        self.scale.lr_multiplier = scale_lr_multiplier
        self.shift.lr_multiplier = shift_lr_multiplier
        
        # Set weight decay as attributes  
        self.scale.weight_decay = scale_weight_decay
        self.shift.weight_decay = shift_weight_decay
        
        # Statistics tracking
        self.register_buffer('scale_stats', torch.zeros(4))  # min, max, mean, std
        self.register_buffer('shift_stats', torch.zeros(4))
        self.register_buffer('activation_stats', torch.zeros(4))
        
    def forward(self, x_hadamard: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable scale and shift to Hadamard-transformed input.
        
        Args:
            x_hadamard: Hadamard-transformed input tensor
            
        Returns:
            Scaled and shifted tensor: γ ⊙ x_hadamard + β
        """
        # Apply constraints if specified
        scale = self.scale
        shift = self.shift
        
        if self.constraint_scale is not None:
            scale = scale.clamp(self.constraint_scale[0], self.constraint_scale[1])
        
        if self.constraint_shift is not None:
            shift = shift.clamp(self.constraint_shift[0], self.constraint_shift[1])
        
        # Apply transformation: γ ⊙ x + β
        result = scale * x_hadamard + shift
        
        # Update statistics during training
        if self.training:
            self._update_statistics(scale, shift, x_hadamard, result)
        
        return result
    
    def _update_statistics(
        self, 
        scale: torch.Tensor, 
        shift: torch.Tensor, 
        x_hadamard: torch.Tensor,
        result: torch.Tensor
    ):
        """Update running statistics for monitoring."""
        with torch.no_grad():
            # Scale statistics
            self.scale_stats[0] = scale.min()
            self.scale_stats[1] = scale.max()
            self.scale_stats[2] = scale.mean()
            self.scale_stats[3] = scale.std()
            
            # Shift statistics
            self.shift_stats[0] = shift.min()
            self.shift_stats[1] = shift.max()
            self.shift_stats[2] = shift.mean()
            self.shift_stats[3] = shift.std()
            
            # Activation statistics
            self.activation_stats[0] = result.min()
            self.activation_stats[1] = result.max()
            self.activation_stats[2] = result.mean()
            self.activation_stats[3] = result.std()
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get current parameter and activation statistics."""
        return {
            'scale': {
                'min': self.scale_stats[0].item(),
                'max': self.scale_stats[1].item(),
                'mean': self.scale_stats[2].item(),
                'std': self.scale_stats[3].item(),
            },
            'shift': {
                'min': self.shift_stats[0].item(),
                'max': self.shift_stats[1].item(),
                'mean': self.shift_stats[2].item(),
                'std': self.shift_stats[3].item(),
            },
            'activations': {
                'min': self.activation_stats[0].item(),
                'max': self.activation_stats[1].item(),
                'mean': self.activation_stats[2].item(),
                'std': self.activation_stats[3].item(),
            },
        }
    
    def reset_parameters(self, init_scale: float = 1.0, init_shift: float = 0.0):
        """Reset learnable parameters to initial values."""
        with torch.no_grad():
            self.scale.fill_(init_scale)
            self.shift.fill_(init_shift)
    
    def extra_repr(self) -> str:
        return (f'size={self.size}, '
                f'scale_lr_mult={getattr(self.scale, "lr_multiplier", 1.0)}, '
                f'shift_lr_mult={getattr(self.shift, "lr_multiplier", 1.0)}')


class AdaptiveHadamardTransform(nn.Module):
    """
    Adaptive Hadamard Transform with Learnable Parameters (AHT-LP) from BitNet v3.
    
    Combines the Hadamard transformation from BitNet v2 with learnable parameters
    that adapt to the activation distributions during training.
    """
    
    def __init__(
        self,
        size: int,
        normalize: bool = True,
        use_fast_transform: bool = True,
        learnable_params: bool = True,
        init_scale: float = 1.0,
        init_shift: float = 0.0,
        scale_lr_multiplier: float = 1.0,
        shift_lr_multiplier: float = 1.0,
        adaptive_init: bool = True,
        momentum_tracking: float = 0.1,
        distribution_aware: bool = True,
    ):
        super().__init__()
        
        self.size = size
        self.normalize = normalize
        self.use_fast_transform = use_fast_transform
        self.learnable_params = learnable_params
        self.adaptive_init = adaptive_init
        self.momentum_tracking = momentum_tracking
        self.distribution_aware = distribution_aware
        
        # Pre-compute Hadamard matrix if needed
        if not use_fast_transform or not is_power_of_2(size):
            if is_power_of_2(size):
                H = create_hadamard_matrix(size)
            else:
                # Pad to next power of 2 and truncate
                padded_size = next_power_of_2(size)
                H_full = create_hadamard_matrix(padded_size)
                H = H_full[:size, :size]
            self.register_buffer('hadamard_matrix', H)
        else:
            self.hadamard_matrix = None
        
        # Learnable parameters
        if learnable_params:
            self.learnable_params_module = LearnableHadamardParameters(
                size=size,
                init_scale=init_scale,
                init_shift=init_shift,
                scale_lr_multiplier=scale_lr_multiplier,
                shift_lr_multiplier=shift_lr_multiplier,
            )
        else:
            # Use fixed scale and shift
            self.register_buffer('scale', torch.ones(size))
            self.register_buffer('shift', torch.zeros(size))
            self.learnable_params_module = None
        
        # Running statistics for adaptive behavior
        if distribution_aware:
            self.register_buffer('running_mean', torch.zeros(size))
            self.register_buffer('running_var', torch.ones(size))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Initialize parameters adaptively if requested
        if adaptive_init and learnable_params:
            self._adaptive_initialization()
    
    def _adaptive_initialization(self):
        """Initialize parameters based on expected activation distributions."""
        with torch.no_grad():
            # Initialize scale based on layer position heuristics
            # Early layers typically need larger scales (1.2-1.5x)
            # Middle layers stay close to 1.0
            # Final layers need smaller scales (0.7-0.9x)
            
            # For now, use a simple random initialization around 1.0
            scale_std = 0.1
            self.learnable_params_module.scale.normal_(1.0, scale_std)
            
            # Initialize shift to small random values
            shift_std = 0.01
            self.learnable_params_module.shift.normal_(0.0, shift_std)
    
    def _apply_hadamard_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the base Hadamard transformation."""
        if self.use_fast_transform and is_power_of_2(self.size):
            return fast_hadamard_transform(x, self.normalize)
        else:
            # Use matrix multiplication
            *batch_dims, size = x.shape
            if size != self.size:
                warnings.warn(f"Input size {size} doesn't match expected size {self.size}")
            
            # Truncate or pad to match expected size
            if size > self.size:
                x = x[..., :self.size]
            elif size < self.size:
                x = F.pad(x, (0, self.size - size))
            
            # Apply transform
            x_flat = x.view(-1, self.size)
            x_transformed = x_flat @ self.hadamard_matrix.t()
            
            if self.normalize:
                x_transformed = x_transformed / math.sqrt(self.size)
            
            return x_transformed.view(*batch_dims, self.size)
    
    def _update_running_stats(self, x: torch.Tensor):
        """Update running statistics for distribution awareness."""
        if not self.distribution_aware or not self.training:
            return
            
        with torch.no_grad():
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics with momentum
            momentum = self.momentum_tracking
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
            
            self.num_batches_tracked += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive Hadamard transform.
        
        Args:
            x: Input tensor of shape (..., size)
            
        Returns:
            Adaptively transformed tensor
        """
        # Apply base Hadamard transformation
        x_hadamard = self._apply_hadamard_transform(x)
        
        # Update running statistics
        self._update_running_stats(x_hadamard)
        
        # Apply learnable parameters
        if self.learnable_params:
            result = self.learnable_params_module(x_hadamard)
        else:
            result = self.scale * x_hadamard + self.shift
        
        return result
    
    def get_learned_patterns(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the learned scale and shift patterns."""
        if not self.learnable_params:
            return None
            
        return {
            'scale': self.learnable_params_module.scale.data.clone(),
            'shift': self.learnable_params_module.shift.data.clone(),
        }
    
    def analyze_adaptation(self) -> Dict[str, Any]:
        """Analyze how the transform has adapted during training."""
        analysis = {
            'size': self.size,
            'num_batches_processed': self.num_batches_tracked.item() if self.distribution_aware else 0,
        }
        
        if self.learnable_params:
            stats = self.learnable_params_module.get_statistics()
            analysis.update(stats)
            
            # Analyze learned patterns
            scale = self.learnable_params_module.scale.data
            shift = self.learnable_params_module.shift.data
            
            analysis['learned_patterns'] = {
                'scale_above_1': (scale > 1.0).sum().item(),
                'scale_below_1': (scale < 1.0).sum().item(),
                'scale_range': (scale.max() - scale.min()).item(),
                'shift_positive': (shift > 0.0).sum().item(),
                'shift_negative': (shift < 0.0).sum().item(),
                'shift_range': (shift.max() - shift.min()).item(),
            }
        
        if self.distribution_aware:
            analysis['distribution_stats'] = {
                'running_mean_norm': self.running_mean.norm().item(),
                'running_var_norm': self.running_var.norm().item(),
                'mean_range': (self.running_mean.max() - self.running_mean.min()).item(),
                'var_range': (self.running_var.max() - self.running_var.min()).item(),
            }
        
        return analysis
    
    def get_parameters_for_optimizer(self, base_lr: float) -> Dict[str, Dict[str, Any]]:
        """Get parameters with their specific learning rates for optimizer setup."""
        if not self.learnable_params:
            return {}
        
        params = {}
        
        # Scale parameters
        scale_lr = base_lr * getattr(self.learnable_params_module.scale, 'lr_multiplier', 1.0)
        params['adaptive_hadamard_scale'] = {
            'params': [self.learnable_params_module.scale],
            'lr': scale_lr,
            'weight_decay': getattr(self.learnable_params_module.scale, 'weight_decay', 0.0),
        }
        
        # Shift parameters
        shift_lr = base_lr * getattr(self.learnable_params_module.shift, 'lr_multiplier', 1.0)
        params['adaptive_hadamard_shift'] = {
            'params': [self.learnable_params_module.shift],
            'lr': shift_lr,
            'weight_decay': getattr(self.learnable_params_module.shift, 'weight_decay', 0.0),
        }
        
        return params
    
    def extra_repr(self) -> str:
        return (f'size={self.size}, normalize={self.normalize}, '
                f'learnable={self.learnable_params}, '
                f'distribution_aware={self.distribution_aware}')


class MultiHeadAdaptiveHadamard(nn.Module):
    """
    Multi-head version of adaptive Hadamard transform for attention mechanisms.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        **kwargs
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Create separate adaptive Hadamard transforms for each head
        self.head_transforms = nn.ModuleList([
            AdaptiveHadamardTransform(self.head_dim, **kwargs)
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head adaptive Hadamard transform.
        
        Args:
            x: Input tensor of shape (..., embed_dim)
            
        Returns:
            Transformed tensor with same shape
        """
        *batch_dims, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {embed_dim}"
        
        # Reshape to separate heads
        x_heads = x.view(*batch_dims, self.num_heads, self.head_dim)
        
        # Apply transform to each head
        transformed_heads = []
        for i, transform in enumerate(self.head_transforms):
            head_input = x_heads[..., i, :]
            transformed_head = transform(head_input)
            transformed_heads.append(transformed_head)
        
        # Concatenate heads back together
        result = torch.stack(transformed_heads, dim=-2)
        return result.view(*batch_dims, embed_dim)
    
    def get_head_analyses(self) -> Dict[int, Dict[str, Any]]:
        """Get analysis for each head."""
        return {i: transform.analyze_adaptation() 
                for i, transform in enumerate(self.head_transforms)}
    
    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}'