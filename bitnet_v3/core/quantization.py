"""
Core quantization functions for BitNet v3.

Implements the quantization schemes described in the BitNet v3 paper:
- Ternary weight quantization {-1, 0, 1} (1.58-bit)
- 4-bit and 8-bit activation quantization
- AbsMean and AbsMax quantization methods
- Progressive quantization for multi-stage training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import math


@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters."""
    weight_bits: float = 1.58
    activation_bits: int = 4
    weight_method: str = "absmean"  # "absmean" or "absmax"
    activation_method: str = "absmax"  # "absmean" or "absmax"
    temperature: float = 1.0
    enable_progressive: bool = True
    

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Round with straight-through estimator.
    Forward: round, Backward: identity
    """
    return (x.round() - x).detach() + x


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Sign with straight-through estimator.
    Forward: sign, Backward: clip gradient to [-1, 1]
    """
    return (x.sign() - x).detach() + x.clamp(-1, 1)


def absmean_quantization(x: torch.Tensor, bits: Union[int, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AbsMean quantization as used in BitNet b1.58.
    
    Args:
        x: Input tensor to quantize
        bits: Number of bits (1.58 for ternary, 4/8 for activations)
        
    Returns:
        Tuple of (quantized_tensor, scale_factor)
    """
    if bits == 1.58:
        # Ternary quantization {-1, 0, 1}
        scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        x_scaled = x / scale
        
        # Ternary quantization with threshold
        x_q = torch.where(
            x_scaled > 0.5, 
            torch.ones_like(x_scaled),
            torch.where(
                x_scaled < -0.5,
                -torch.ones_like(x_scaled),
                torch.zeros_like(x_scaled)
            )
        )
        return x_q * scale, scale
    else:
        # Multi-bit quantization for activations
        n_levels = 2 ** bits
        scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        x_scaled = x / scale
        x_q = round_ste(x_scaled.clamp(-n_levels/2, n_levels/2 - 1))
        return x_q * scale, scale


def absmax_quantization(x: torch.Tensor, bits: Union[int, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AbsMax quantization for per-token quantization.
    
    Args:
        x: Input tensor to quantize
        bits: Number of bits for quantization
        
    Returns:
        Tuple of (quantized_tensor, scale_factor)
    """
    if bits == 1.58:
        # Ternary quantization with absmax scaling
        scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        x_scaled = x / scale
        
        x_q = torch.where(
            x_scaled > 0.5,
            torch.ones_like(x_scaled),
            torch.where(
                x_scaled < -0.5,
                -torch.ones_like(x_scaled),
                torch.zeros_like(x_scaled)
            )
        )
        return x_q * scale, scale
    else:
        # Multi-bit quantization
        n_levels = 2 ** bits
        scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        x_scaled = x / scale
        x_q = round_ste(x_scaled.clamp(-n_levels/2, n_levels/2 - 1))
        return x_q * scale, scale


def quantize_weights_158(weight: torch.Tensor, method: str = "absmean") -> torch.Tensor:
    """
    Quantize weights to 1.58-bit (ternary) as in BitNet v3.
    
    Args:
        weight: Weight tensor to quantize
        method: Quantization method ("absmean" or "absmax")
        
    Returns:
        Quantized weight tensor
    """
    if method == "absmean":
        quantized, _ = absmean_quantization(weight, 1.58)
    elif method == "absmax":
        quantized, _ = absmax_quantization(weight, 1.58)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    return quantized


def quantize_activations(
    x: torch.Tensor, 
    bits: Union[int, float] = 4, 
    method: str = "absmax",
    per_token: bool = True
) -> torch.Tensor:
    """
    Quantize activations to specified bit-width.
    
    Args:
        x: Input activation tensor
        bits: Number of bits (4, 8, 16, or float for FP16)
        method: Quantization method ("absmean" or "absmax")
        per_token: Whether to use per-token quantization
        
    Returns:
        Quantized activation tensor
    """
    if bits == 16 or bits >= 16:
        # Full precision - no quantization
        return x
    elif bits == 8:
        # 8-bit activations use per-token absmax
        quantized, _ = absmax_quantization(x, bits)
    elif bits == 4:
        # 4-bit activations use per-token absmean (as in BitNet v2)
        if method == "absmean":
            quantized, _ = absmean_quantization(x, bits)
        else:
            quantized, _ = absmax_quantization(x, bits)
    elif bits == 2:
        # 2-bit quantization
        quantized, _ = absmax_quantization(x, bits)
    else:
        raise ValueError(f"Unsupported activation bits: {bits}")
    
    return quantized


def progressive_quantize(
    x: torch.Tensor,
    current_bits: Union[int, float],
    target_bits: Union[int, float], 
    temperature: float,
    is_weight: bool = True
) -> torch.Tensor:
    """
    Progressive quantization with temperature-based transition.
    
    Implements the equation from BitNet v3 paper:
    Q_t(x) = σ(β_t) · Q_{b_t}(x) + (1 - σ(β_t)) · Q_{b_{t-1}}(x)
    
    Args:
        x: Input tensor
        current_bits: Current quantization level
        target_bits: Target quantization level
        temperature: Temperature parameter β_t
        is_weight: Whether this is a weight (vs activation)
        
    Returns:
        Progressively quantized tensor
    """
    # Compute mixing weight using sigmoid
    alpha = torch.sigmoid(torch.tensor(temperature))
    
    # Quantize at both levels
    if is_weight:
        if current_bits == 1.58:
            x_current = quantize_weights_158(x)
        elif current_bits >= 16:
            x_current = x  # No quantization for FP16
        else:
            x_current = quantize_activations(x, current_bits)
            
        if target_bits == 1.58:
            x_target = quantize_weights_158(x)
        elif target_bits >= 16:
            x_target = x  # No quantization for FP16
        else:
            x_target = quantize_activations(x, target_bits)
    else:
        x_current = quantize_activations(x, current_bits)
        x_target = quantize_activations(x, target_bits)
    
    # Linear interpolation between quantization levels
    return alpha * x_target + (1 - alpha) * x_current


class QuantizationFunction(torch.autograd.Function):
    """
    Custom autograd function for quantization with straight-through estimator.
    """
    
    @staticmethod
    def forward(ctx, x, bits, method="absmean", is_weight=True):
        if is_weight:
            return quantize_weights_158(x, method)
        else:
            return quantize_activations(x, bits, method)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients through unchanged
        return grad_output, None, None, None


class AdaptiveQuantization(nn.Module):
    """
    Adaptive quantization module that adjusts based on input statistics.
    """
    
    def __init__(
        self,
        bits: Union[int, float] = 1.58,
        method: str = "absmean",
        is_weight: bool = True,
        adaptive_threshold: bool = True,
        momentum: float = 0.1
    ):
        super().__init__()
        self.bits = bits
        self.method = method
        self.is_weight = is_weight
        self.adaptive_threshold = adaptive_threshold
        self.momentum = momentum
        
        # Running statistics for adaptive thresholding
        if adaptive_threshold:
            self.register_buffer('running_mean', torch.zeros(1))
            self.register_buffer('running_var', torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptive_threshold and self.training:
            # Update running statistics
            batch_mean = x.mean()
            batch_var = x.var()
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        
        # Apply quantization
        return QuantizationFunction.apply(x, self.bits, self.method, self.is_weight)


class BitNetQuantizer(nn.Module):
    """
    Unified quantizer for BitNet v3 with all quantization schemes.
    """
    
    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        
        # Weight quantizer
        self.weight_quantizer = AdaptiveQuantization(
            bits=config.weight_bits,
            method=config.weight_method,
            is_weight=True
        )
        
        # Activation quantizer
        self.activation_quantizer = AdaptiveQuantization(
            bits=config.activation_bits,
            method=config.activation_method,
            is_weight=False
        )
    
    def quantize_weights(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weight tensor."""
        return self.weight_quantizer(weight)
    
    def quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation tensor."""
        return self.activation_quantizer(x)
    
    def progressive_quantize_weights(
        self, 
        weight: torch.Tensor,
        current_bits: Union[int, float],
        target_bits: Union[int, float],
        temperature: float
    ) -> torch.Tensor:
        """Apply progressive quantization to weights."""
        return progressive_quantize(weight, current_bits, target_bits, temperature, is_weight=True)
    
    def progressive_quantize_activations(
        self,
        x: torch.Tensor,
        current_bits: int,
        target_bits: int,
        temperature: float
    ) -> torch.Tensor:
        """Apply progressive quantization to activations."""
        return progressive_quantize(x, current_bits, target_bits, temperature, is_weight=False)


def compute_quantization_error(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """
    Compute quantization error for regularization purposes.
    
    Args:
        original: Original tensor
        quantized: Quantized tensor
        
    Returns:
        L2 quantization error
    """
    return torch.norm(original - quantized, p=2, dim=-1).mean()


def get_layer_sensitivity(gradients: torch.Tensor) -> torch.Tensor:
    """
    Compute layer sensitivity for dynamic regularization.
    
    Args:
        gradients: Gradient tensor for the layer
        
    Returns:
        Sensitivity weight for the layer
    """
    return torch.norm(gradients, p=2)