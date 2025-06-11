"""
Core BitNet v3 components including quantization functions, Hadamard transforms, and gradient estimators.

This module provides the fundamental building blocks for BitNet v3:
- Quantization functions for weights and activations
- Hadamard transform utilities  
- Straight-through estimators for gradient approximation
"""

from .quantization import (
    quantize_weights_158,
    quantize_activations,
    absmean_quantization,
    absmax_quantization,
    progressive_quantize,
    QuantizationConfig,
)

from .hadamard import (
    hadamard_transform,
    create_hadamard_matrix,
    fast_hadamard_transform,
    batch_hadamard_transform,
)

from .straight_through import (
    StraightThroughEstimator,
    EnhancedSTEWithMomentum,
    MomentumGradientBuffer,
)

__all__ = [
    # Quantization
    "quantize_weights_158",
    "quantize_activations",
    "absmean_quantization", 
    "absmax_quantization",
    "progressive_quantize",
    "QuantizationConfig",
    
    # Hadamard transforms
    "hadamard_transform",
    "create_hadamard_matrix",
    "fast_hadamard_transform",
    "batch_hadamard_transform",
    
    # Straight-through estimators
    "StraightThroughEstimator",
    "EnhancedSTEWithMomentum",
    "MomentumGradientBuffer",
]