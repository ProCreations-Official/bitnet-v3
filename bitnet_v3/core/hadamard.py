"""
Hadamard transform utilities for BitNet v3.

Implements the Hadamard transformation used in BitNet v2 and enhanced in BitNet v3
with adaptive learnable parameters. The Hadamard transform helps smooth activation
distributions for better quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import warnings


def is_power_of_2(n: int) -> bool:
    """Check if a number is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_2(n: int) -> int:
    """Find the next power of 2 greater than or equal to n."""
    if n <= 1:
        return 1
    return 2 ** (n - 1).bit_length()


def create_hadamard_matrix(size: int) -> torch.Tensor:
    """
    Create a Hadamard matrix of given size.
    
    Args:
        size: Size of the Hadamard matrix (must be power of 2)
        
    Returns:
        Hadamard matrix of shape (size, size)
    """
    if not is_power_of_2(size):
        raise ValueError(f"Size must be a power of 2, got {size}")
    
    if size == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    
    # Recursive construction using Sylvester's construction
    half_size = size // 2
    H_half = create_hadamard_matrix(half_size)
    
    # Construct full matrix
    H = torch.zeros(size, size, dtype=torch.float32)
    H[:half_size, :half_size] = H_half
    H[:half_size, half_size:] = H_half
    H[half_size:, :half_size] = H_half
    H[half_size:, half_size:] = -H_half
    
    return H


def hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Apply Hadamard transform to input tensor.
    
    Args:
        x: Input tensor of shape (..., size) where size must be power of 2
        normalize: Whether to normalize by sqrt(size)
        
    Returns:
        Hadamard transformed tensor
    """
    *batch_dims, size = x.shape
    
    if not is_power_of_2(size):
        # Pad to next power of 2
        padded_size = next_power_of_2(size)
        x_padded = F.pad(x, (0, padded_size - size))
        result = fast_hadamard_transform(x_padded, normalize)
        return result[..., :size]
    
    return fast_hadamard_transform(x, normalize)


def fast_hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Fast Hadamard Transform using the Fast Walsh-Hadamard Transform algorithm.
    
    Args:
        x: Input tensor (..., size) where size is power of 2
        normalize: Whether to normalize by sqrt(size)
        
    Returns:
        Transformed tensor
    """
    *batch_dims, size = x.shape
    
    if not is_power_of_2(size):
        raise ValueError(f"Input size must be power of 2, got {size}")
    
    # Reshape for matrix operations
    x_flat = x.view(-1, size)
    batch_size = x_flat.size(0)
    
    # Apply fast transform
    result = x_flat.clone()
    h = 1
    while h < size:
        for i in range(0, size, h * 2):
            for j in range(h):
                u = result[:, i + j]
                v = result[:, i + j + h]
                result[:, i + j] = u + v
                result[:, i + j + h] = u - v
        h *= 2
    
    if normalize:
        result = result / math.sqrt(size)
    
    # Reshape back to original shape
    return result.view(*batch_dims, size)


def batch_hadamard_transform(
    x: torch.Tensor, 
    hadamard_matrix: Optional[torch.Tensor] = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Apply Hadamard transform using matrix multiplication (slower but more flexible).
    
    Args:
        x: Input tensor of shape (..., size)
        hadamard_matrix: Pre-computed Hadamard matrix. If None, will be created.
        normalize: Whether to normalize the transform
        
    Returns:
        Transformed tensor
    """
    *batch_dims, size = x.shape
    
    if hadamard_matrix is None:
        # Create or pad Hadamard matrix
        if is_power_of_2(size):
            H = create_hadamard_matrix(size)
        else:
            padded_size = next_power_of_2(size)
            H = create_hadamard_matrix(padded_size)[:size, :size]
    else:
        H = hadamard_matrix
    
    # Move to same device and dtype as input
    H = H.to(device=x.device, dtype=x.dtype)
    
    # Apply transform: x @ H.T
    x_flat = x.view(-1, size)
    result = x_flat @ H.t()
    
    if normalize:
        result = result / math.sqrt(size)
    
    return result.view(*batch_dims, size)


class HadamardTransform(nn.Module):
    """
    Hadamard transform layer for neural networks.
    """
    
    def __init__(
        self, 
        size: int, 
        normalize: bool = True,
        learnable_scale: bool = False,
        learnable_shift: bool = False,
        use_fast_transform: bool = True
    ):
        super().__init__()
        self.size = size
        self.normalize = normalize
        self.use_fast_transform = use_fast_transform
        
        # Pre-compute Hadamard matrix if not using fast transform
        if not use_fast_transform:
            if is_power_of_2(size):
                H = create_hadamard_matrix(size)
            else:
                padded_size = next_power_of_2(size)
                H = create_hadamard_matrix(padded_size)[:size, :size]
            self.register_buffer('hadamard_matrix', H)
        else:
            self.hadamard_matrix = None
        
        # Learnable parameters for adaptive transform (BitNet v3)
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(size))
        else:
            self.register_buffer('scale', torch.ones(size))
            
        if learnable_shift:
            self.shift = nn.Parameter(torch.zeros(size))
        else:
            self.register_buffer('shift', torch.zeros(size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hadamard transform to input.
        
        Args:
            x: Input tensor of shape (..., size)
            
        Returns:
            Transformed tensor
        """
        if self.use_fast_transform and is_power_of_2(self.size):
            x_transformed = fast_hadamard_transform(x, self.normalize)
        else:
            x_transformed = batch_hadamard_transform(
                x, self.hadamard_matrix, self.normalize
            )
        
        # Apply learnable scale and shift (adaptive Hadamard in BitNet v3)
        return self.scale * x_transformed + self.shift
    
    def extra_repr(self) -> str:
        return f'size={self.size}, normalize={self.normalize}, use_fast={self.use_fast_transform}'


class AdaptiveHadamardTransform(nn.Module):
    """
    Adaptive Hadamard Transform with Learnable Parameters (AHT-LP) from BitNet v3.
    
    Implements the equation: H_adaptive(x) = γ ⊙ (H_m · x) + β
    where γ and β are learnable parameters.
    """
    
    def __init__(
        self,
        size: int,
        normalize: bool = True,
        init_scale: float = 1.0,
        init_shift: float = 0.0,
        scale_lr_multiplier: float = 1.0,
        shift_lr_multiplier: float = 1.0,
        use_fast_transform: bool = True
    ):
        super().__init__()
        self.size = size
        self.normalize = normalize
        self.use_fast_transform = use_fast_transform
        
        # Learnable scale (γ) and shift (β) parameters
        self.scale = nn.Parameter(torch.full((size,), init_scale))
        self.shift = nn.Parameter(torch.full((size,), init_shift))
        
        # Learning rate multipliers (as mentioned in paper)
        self.scale.lr_multiplier = scale_lr_multiplier
        self.shift.lr_multiplier = shift_lr_multiplier
        
        # Pre-compute Hadamard matrix if needed
        if not use_fast_transform or not is_power_of_2(size):
            if is_power_of_2(size):
                H = create_hadamard_matrix(size)
            else:
                padded_size = next_power_of_2(size)
                H = create_hadamard_matrix(padded_size)[:size, :size]
            self.register_buffer('hadamard_matrix', H)
        else:
            self.hadamard_matrix = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive Hadamard transform.
        
        Args:
            x: Input tensor of shape (..., size)
            
        Returns:
            Adaptively transformed tensor
        """
        # Apply base Hadamard transform
        if self.use_fast_transform and is_power_of_2(self.size):
            x_hadamard = fast_hadamard_transform(x, self.normalize)
        else:
            x_hadamard = batch_hadamard_transform(
                x, self.hadamard_matrix, self.normalize
            )
        
        # Apply learnable scale and shift: γ ⊙ (H_m · x) + β
        return self.scale * x_hadamard + self.shift
    
    def get_parameters_for_lr_scaling(self):
        """Return parameters that need learning rate scaling."""
        return {
            'scale': (self.scale, getattr(self.scale, 'lr_multiplier', 1.0)),
            'shift': (self.shift, getattr(self.shift, 'lr_multiplier', 1.0)),
        }
    
    def extra_repr(self) -> str:
        return (f'size={self.size}, normalize={self.normalize}, '
                f'scale_init={self.scale.data[0].item():.3f}, '
                f'shift_init={self.shift.data[0].item():.3f}')


def test_hadamard_transform():
    """Test function to verify Hadamard transform implementation."""
    # Test sizes
    sizes = [1, 2, 4, 8, 16, 32, 64]
    
    for size in sizes:
        print(f"Testing size {size}...")
        
        # Create test input
        x = torch.randn(2, 3, size)
        
        # Test fast transform
        if is_power_of_2(size):
            y_fast = fast_hadamard_transform(x)
            
            # Test matrix-based transform
            H = create_hadamard_matrix(size)
            y_matrix = batch_hadamard_transform(x, H)
            
            # Should be approximately equal
            assert torch.allclose(y_fast, y_matrix, atol=1e-5), f"Mismatch at size {size}"
        
        # Test general transform
        y_general = hadamard_transform(x)
        
        print(f"Size {size}: ✓")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_hadamard_transform()