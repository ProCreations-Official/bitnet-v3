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
    ternary_threshold: float = 0.5
    ternary_neg_val_init: float = 1.0
    ternary_pos_val_init: float = 1.0
    activation_learnable_step_init: float = 1.0
    round_ste_grad_scale_factor: float = 1.0
    scale_learnable_multiplier_init: float = 1.0
    enable_scale_learnable_multiplier: bool = False
    quantization_error_method: str = "l2"
    quantization_error_moment_weights: Tuple[float, float] = (1.0, 1.0)


class ScaledRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, grad_scale_factor: float):
        ctx.grad_scale_factor = grad_scale_factor
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output * ctx.grad_scale_factor, None


def round_ste(x: torch.Tensor, grad_scale_factor: float = 1.0) -> torch.Tensor:
    """
    Round with straight-through estimator.
    Forward: round, Backward: identity * grad_scale_factor
    """
    if abs(grad_scale_factor - 1.0) < 1e-7: # Check if it's effectively 1.0
        return (x.round() - x).detach() + x
    else:
        return ScaledRoundSTE.apply(x, grad_scale_factor)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Sign with straight-through estimator.
    Forward: sign, Backward: clip gradient to [-1, 1]
    """
    return (x.sign() - x).detach() + x.clamp(-1, 1)


def absmean_quantization(x: torch.Tensor, bits: Union[int, float], ternary_threshold: float = 0.5, quant_neg_val: Optional[torch.Tensor] = None, quant_pos_val: Optional[torch.Tensor] = None, learnable_step: Optional[torch.Tensor] = None, grad_scale_factor: float = 1.0, scale_multiplier: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AbsMean quantization as used in BitNet b1.58.
    
    Args:
        x: Input tensor to quantize
        bits: Number of bits (1.58 for ternary, 4/8 for activations)
        ternary_threshold: Threshold for ternary quantization
        quant_neg_val: Negative value for ternary quantization (e.g., learnable)
        quant_pos_val: Positive value for ternary quantization (e.g., learnable)
        learnable_step: Learnable step size for activation quantization
        grad_scale_factor: Gradient scaling factor for round_ste
        scale_multiplier: Learnable multiplier for the data-driven scale
        
    Returns:
        Tuple of (quantized_tensor, scale_factor)
    """
    if bits == 1.58:
        # Ternary quantization {-1, 0, 1} or {-quant_neg_val, 0, quant_pos_val}
        scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        if scale_multiplier is not None:
            scale = scale * scale_multiplier
        x_scaled = x / scale
        
        _quant_pos_val = quant_pos_val if quant_pos_val is not None else torch.ones_like(x_scaled)
        _quant_neg_val = quant_neg_val if quant_neg_val is not None else torch.ones_like(x_scaled)

        # Ternary quantization with threshold
        x_q = torch.where(
            x_scaled > ternary_threshold,
            _quant_pos_val.expand_as(x_scaled),
            torch.where(
                x_scaled < -ternary_threshold,
                -_quant_neg_val.expand_as(x_scaled),
                torch.zeros_like(x_scaled)
            )
        )
        return x_q * scale, scale
    else:
        # Multi-bit quantization for activations
        scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        if scale_multiplier is not None:
            scale = scale * scale_multiplier
        if learnable_step is not None:
            n_levels = 2 ** bits
            Q_max = n_levels / 2
            x_scaled_by_global_scale = x / scale # x_scaled by global scale, not learnable step yet
            x_quant_normalized = round_ste(torch.clamp(x_scaled_by_global_scale / learnable_step, -Q_max, Q_max - 1), grad_scale_factor=grad_scale_factor)
            x_q = x_quant_normalized * learnable_step
        else:
            n_levels = 2 ** bits
            x_scaled = x / scale
            x_q = round_ste(x_scaled.clamp(-n_levels/2, n_levels/2 - 1), grad_scale_factor=grad_scale_factor)
        return x_q * scale, scale


def absmax_quantization(x: torch.Tensor, bits: Union[int, float], ternary_threshold: float = 0.5, quant_neg_val: Optional[torch.Tensor] = None, quant_pos_val: Optional[torch.Tensor] = None, learnable_step: Optional[torch.Tensor] = None, grad_scale_factor: float = 1.0, scale_multiplier: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AbsMax quantization for per-token quantization.
    
    Args:
        x: Input tensor to quantize
        bits: Number of bits for quantization
        ternary_threshold: Threshold for ternary quantization
        quant_neg_val: Negative value for ternary quantization (e.g., learnable)
        quant_pos_val: Positive value for ternary quantization (e.g., learnable)
        learnable_step: Learnable step size for activation quantization
        grad_scale_factor: Gradient scaling factor for round_ste
        scale_multiplier: Learnable multiplier for the data-driven scale
        
    Returns:
        Tuple of (quantized_tensor, scale_factor)
    """
    if bits == 1.58:
        # Ternary quantization with absmax scaling
        scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        if scale_multiplier is not None:
            scale = scale * scale_multiplier
        x_scaled = x / scale

        _quant_pos_val = quant_pos_val if quant_pos_val is not None else torch.ones_like(x_scaled)
        _quant_neg_val = quant_neg_val if quant_neg_val is not None else torch.ones_like(x_scaled)
        
        x_q = torch.where(
            x_scaled > ternary_threshold,
            _quant_pos_val.expand_as(x_scaled),
            torch.where(
                x_scaled < -ternary_threshold,
                -_quant_neg_val.expand_as(x_scaled),
                torch.zeros_like(x_scaled)
            )
        )
        return x_q * scale, scale
    else:
        # Multi-bit quantization
        scale = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        if scale_multiplier is not None:
            scale = scale * scale_multiplier
        if learnable_step is not None:
            n_levels = 2 ** bits
            Q_max = n_levels / 2
            x_scaled_by_global_scale = x / scale # x_scaled by global scale, not learnable step yet
            x_quant_normalized = round_ste(torch.clamp(x_scaled_by_global_scale / learnable_step, -Q_max, Q_max - 1), grad_scale_factor=grad_scale_factor)
            x_q = x_quant_normalized * learnable_step
        else:
            n_levels = 2 ** bits
            x_scaled = x / scale
            x_q = round_ste(x_scaled.clamp(-n_levels/2, n_levels/2 - 1), grad_scale_factor=grad_scale_factor)
        return x_q * scale, scale


def quantize_weights_158(weight: torch.Tensor, method: str = "absmean", ternary_threshold: float = 0.5, quant_neg_val: Optional[torch.Tensor] = None, quant_pos_val: Optional[torch.Tensor] = None, scale_multiplier: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Quantize weights to 1.58-bit (ternary) as in BitNet v3.
    
    Args:
        weight: Weight tensor to quantize
        method: Quantization method ("absmean" or "absmax")
        ternary_threshold: Threshold for ternary quantization
        quant_neg_val: Negative value for ternary quantization
        quant_pos_val: Positive value for ternary quantization
        scale_multiplier: Learnable multiplier for the data-driven scale
        
    Returns:
        Quantized weight tensor
    """
    if method == "absmean":
        quantized, _ = absmean_quantization(weight, 1.58, ternary_threshold=ternary_threshold, quant_neg_val=quant_neg_val, quant_pos_val=quant_pos_val, scale_multiplier=scale_multiplier)
    elif method == "absmax":
        quantized, _ = absmax_quantization(weight, 1.58, ternary_threshold=ternary_threshold, quant_neg_val=quant_neg_val, quant_pos_val=quant_pos_val, scale_multiplier=scale_multiplier)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    return quantized


def quantize_activations(
    x: torch.Tensor, 
    bits: Union[int, float] = 4, 
    method: str = "absmax",
    per_token: bool = True, # Note: per_token is not explicitly used in current absmean/absmax for activations after LSQ introduction logic
    learnable_step: Optional[torch.Tensor] = None,
    grad_scale_factor: float = 1.0,
    scale_multiplier: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Quantize activations to specified bit-width.
    
    Args:
        x: Input activation tensor
        bits: Number of bits (4, 8, 16, or float for FP16)
        method: Quantization method ("absmean" or "absmax")
        per_token: Whether to use per-token quantization
        learnable_step: Learnable step size for LSQ-like quantization
        grad_scale_factor: Gradient scaling factor for round_ste in activation quantization
        scale_multiplier: Learnable multiplier for the data-driven scale
        
    Returns:
        Quantized activation tensor
    """
    if bits == 16 or bits >= 16:
        # Full precision - no quantization
        return x
    elif bits == 8:
        # 8-bit activations use per-token absmax
        quantized, _ = absmax_quantization(x, bits, learnable_step=learnable_step, grad_scale_factor=grad_scale_factor, scale_multiplier=scale_multiplier)
    elif bits == 4:
        # 4-bit activations use per-token absmean (as in BitNet v2)
        if method == "absmean":
            quantized, _ = absmean_quantization(x, bits, learnable_step=learnable_step, grad_scale_factor=grad_scale_factor, scale_multiplier=scale_multiplier)
        else:
            quantized, _ = absmax_quantization(x, bits, learnable_step=learnable_step, grad_scale_factor=grad_scale_factor, scale_multiplier=scale_multiplier)
    elif bits == 2:
        # 2-bit quantization
        quantized, _ = absmax_quantization(x, bits, learnable_step=learnable_step, grad_scale_factor=grad_scale_factor, scale_multiplier=scale_multiplier)
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
    def forward(ctx, x, bits, method="absmean", is_weight=True, ternary_threshold=0.5, quant_neg_val=None, quant_pos_val=None, learnable_step=None, round_ste_grad_scale_factor=1.0, scale_multiplier=None):
        if is_weight:
            # learnable_step and round_ste_grad_scale_factor are not used for weights currently
            if bits == 1.58:
                return quantize_weights_158(x, method, ternary_threshold=ternary_threshold, quant_neg_val=quant_neg_val, quant_pos_val=quant_pos_val, scale_multiplier=scale_multiplier)
            else:
                # Assuming non-1.58 bit weights might also use scale_multiplier.
                # If not, this should be conditional.
                return quantize_weights_158(x, method, ternary_threshold=ternary_threshold, scale_multiplier=scale_multiplier)
        else:
            # quant_neg_val and quant_pos_val are not used for activations directly here
            return quantize_activations(x, bits, method, learnable_step=learnable_step, grad_scale_factor=round_ste_grad_scale_factor, scale_multiplier=scale_multiplier)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients through unchanged
        return grad_output, None, None, None, None, None, None, None, None, None # Added None for scale_multiplier


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
        momentum: float = 0.1,
        ternary_threshold: float = 0.5,
        ternary_neg_val_init: float = 1.0,
        ternary_pos_val_init: float = 1.0,
        activation_learnable_step_init: float = 1.0,
        round_ste_grad_scale_factor: float = 1.0,
        scale_learnable_multiplier_init: float = 1.0,
        enable_scale_learnable_multiplier: bool = False
    ):
        super().__init__()
        self.bits = bits
        self.method = method
        self.is_weight = is_weight
        self.adaptive_threshold = adaptive_threshold
        self.momentum = momentum
        self.activation_learnable_step = None
        self.round_ste_grad_scale_factor = round_ste_grad_scale_factor
        self.enable_scale_learnable_multiplier = enable_scale_learnable_multiplier

        if self.enable_scale_learnable_multiplier:
            self.scale_multiplier = torch.nn.Parameter(torch.tensor(float(scale_learnable_multiplier_init)))
        else:
            self.scale_multiplier = None

        if self.is_weight and self.bits == 1.58:
            self.learnable_ternary_threshold = torch.nn.Parameter(torch.tensor(float(ternary_threshold)))
            self.quant_neg_val = torch.nn.Parameter(torch.tensor(float(ternary_neg_val_init)))
            self.quant_pos_val = torch.nn.Parameter(torch.tensor(float(ternary_pos_val_init)))
            self.ternary_threshold = None
        elif not self.is_weight and self.bits < 16: # Quantizable activations
            self.activation_learnable_step = torch.nn.Parameter(torch.tensor(float(activation_learnable_step_init)))
            self.ternary_threshold = ternary_threshold
            self.learnable_ternary_threshold = None
            self.quant_neg_val = None
            self.quant_pos_val = None
        else: # Non-1.58bit weights or non-quantizable activations (FP16/BF16)
            self.ternary_threshold = ternary_threshold
            self.learnable_ternary_threshold = None
            self.quant_neg_val = None
            self.quant_pos_val = None
        
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
        current_scale_multiplier = self.scale_multiplier if self.enable_scale_learnable_multiplier else None

        if self.is_weight and self.bits == 1.58:
            # Pass None for learnable_step and default for round_ste_grad_scale_factor (as it's not used for weights)
            return QuantizationFunction.apply(x, self.bits, self.method, self.is_weight, self.learnable_ternary_threshold, self.quant_neg_val, self.quant_pos_val, None, 1.0, current_scale_multiplier)
        elif not self.is_weight and self.bits < 16: # Quantizable activations
             # Pass None for weight-specific params, pass activation_learnable_step and round_ste_grad_scale_factor
             return QuantizationFunction.apply(x, self.bits, self.method, self.is_weight, self.ternary_threshold if self.ternary_threshold is not None else 0.5, None, None, self.activation_learnable_step, self.round_ste_grad_scale_factor, current_scale_multiplier)
        else: # Non-1.58bit weights or non-quantized activations (e.g. FP16)
            # Pass all relevant stored params; QuantizationFunction.forward will select appropriately
            return QuantizationFunction.apply(x, self.bits, self.method, self.is_weight, self.ternary_threshold if self.ternary_threshold is not None else 0.5, self.quant_neg_val, self.quant_pos_val, self.activation_learnable_step, self.round_ste_grad_scale_factor, current_scale_multiplier)


class BitNetQuantizer(nn.Module):
    """
    Unified quantizer for BitNet v3 with all quantization schemes.
    """
    
    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.ternary_threshold = getattr(config, 'ternary_threshold', 0.5)
        self.ternary_neg_val_init = getattr(config, 'ternary_neg_val_init', 1.0)
        self.ternary_pos_val_init = getattr(config, 'ternary_pos_val_init', 1.0)
        self.activation_learnable_step_init = getattr(config, 'activation_learnable_step_init', 1.0)
        self.round_ste_grad_scale_factor = getattr(config, 'round_ste_grad_scale_factor', 1.0)
        self.scale_learnable_multiplier_init = getattr(config, 'scale_learnable_multiplier_init', 1.0)
        self.enable_scale_learnable_multiplier = getattr(config, 'enable_scale_learnable_multiplier', False)
        
        # Weight quantizer
        weight_adaptive_args = {
            "bits": config.weight_bits,
            "method": config.weight_method,
            "is_weight": True,
            "ternary_threshold": self.ternary_threshold,
            "scale_learnable_multiplier_init": self.scale_learnable_multiplier_init,
            "enable_scale_learnable_multiplier": self.enable_scale_learnable_multiplier
        }
        if config.weight_bits == 1.58:
            weight_adaptive_args["ternary_neg_val_init"] = self.ternary_neg_val_init
            weight_adaptive_args["ternary_pos_val_init"] = self.ternary_pos_val_init
        self.weight_quantizer = AdaptiveQuantization(**weight_adaptive_args)
        
        # Activation quantizer
        activation_adaptive_args = {
            "bits": config.activation_bits,
            "method": config.activation_method,
            "is_weight": False,
            "activation_learnable_step_init": self.activation_learnable_step_init,
            "round_ste_grad_scale_factor": self.round_ste_grad_scale_factor,
            "scale_learnable_multiplier_init": self.scale_learnable_multiplier_init,
            "enable_scale_learnable_multiplier": self.enable_scale_learnable_multiplier
        }
        self.activation_quantizer = AdaptiveQuantization(**activation_adaptive_args)
    
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


def compute_quantization_error(original: torch.Tensor, quantized: torch.Tensor, method: str = "l2", moment_weights: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Compute quantization error for regularization purposes.
    
    Args:
        original: Original tensor
        quantized: Quantized tensor
        method: Error computation method ("l1", "l2", "moments")
        moment_weights: Tuple of weights (w_mean, w_var) for "moments" method
        
    Returns:
        Computed quantization error
    """
    if method == "l1":
        return torch.norm(original - quantized, p=1, dim=-1).mean()
    elif method == "moments":
        mean_orig = original.mean(dim=-1)
        mean_quant = quantized.mean(dim=-1)
        var_orig = original.var(dim=-1)
        var_quant = quantized.var(dim=-1)

        mean_abs_diff = (mean_orig - mean_quant).abs().mean()
        var_abs_diff = (var_orig - var_quant).abs().mean()

        w_mean, w_var = moment_weights if moment_weights else (1.0, 1.0)
        return w_mean * mean_abs_diff + w_var * var_abs_diff
    elif method == "l2":
        # Default L2
        return torch.norm(original - quantized, p=2, dim=-1).mean()
    else:
        raise ValueError(f"Unknown quantization error method: {method}")


def get_layer_sensitivity(gradients: torch.Tensor) -> torch.Tensor:
    """
    Compute layer sensitivity for dynamic regularization.
    
    Args:
        gradients: Gradient tensor for the layer
        
    Returns:
        Sensitivity weight for the layer
    """
    return torch.norm(gradients, p=2)


def pack_ternary_weights(quantized_weights: torch.Tensor, true_positive_value: float, true_negative_value: float, tolerance: float = 1e-4) -> torch.Tensor:
    """
    Packs a ternary weight tensor into a torch.uint8 tensor.
    Assumes weights are close to true_positive_value, true_negative_value, or 0.
    Mapping: 0 -> 00_bin, 1 (positive) -> 01_bin, 2 (negative) -> 10_bin.
    Four such 2-bit values are packed into a single uint8 byte.
    """
    int_weights = torch.zeros_like(quantized_weights, dtype=torch.uint8)

    # Create masks first to handle potential overlaps if tolerance is large or values are close
    is_pos = torch.abs(quantized_weights - true_positive_value) < tolerance
    is_neg = torch.abs(quantized_weights - true_negative_value) < tolerance
    # Assign zero to remaining elements, more robustly than direct comparison if values are not exact
    is_zero = ~(is_pos | is_neg)

    int_weights[is_zero] = 0  # Represents 0.0
    int_weights[is_pos] = 1  # Represents true_positive_value
    int_weights[is_neg] = 2  # Represents true_negative_value

    flattened_weights = int_weights.flatten()
    original_numel = flattened_weights.numel()

    # Pad with zeros if numel is not divisible by 4
    padding_size = (4 - (original_numel % 4)) % 4
    if padding_size > 0:
        flattened_weights = torch.cat([flattened_weights, torch.zeros(padding_size, dtype=torch.uint8, device=quantized_weights.device)])

    reshaped_weights = flattened_weights.view(-1, 4)

    packed_bytes = torch.zeros(reshaped_weights.shape[0], dtype=torch.uint8, device=quantized_weights.device)

    # Pack 4 integers (2 bits each) into one uint8 byte
    # s0 (00_bin), s1 (01_bin), s2 (10_bin), s3 (11_bin - unused but fits)
    # byte = (s3 << 6) | (s2 << 4) | (s1 << 2) | (s0 << 0)
    packed_bytes = (reshaped_weights[:, 3] << 6) | \
                   (reshaped_weights[:, 2] << 4) | \
                   (reshaped_weights[:, 1] << 2) | \
                   (reshaped_weights[:, 0] << 0)

    return packed_bytes


def unpack_ternary_weights(packed_weights: torch.Tensor, original_shape: torch.Size, true_positive_value: float, true_negative_value: float) -> torch.Tensor:
    """
    Unpacks a torch.uint8 tensor into a ternary float tensor.
    """
    packed_flat = packed_weights.flatten()
    num_bytes = packed_flat.numel()

    int_values_list = []

    for i in range(num_bytes):
        byte = packed_flat[i].item()
        s0 = (byte >> 0) & 0x03  # Extracts bits 0-1
        s1 = (byte >> 2) & 0x03  # Extracts bits 2-3
        s2 = (byte >> 4) & 0x03  # Extracts bits 4-5
        s3 = (byte >> 6) & 0x03  # Extracts bits 6-7
        int_values_list.extend([s0, s1, s2, s3])

    int_tensor = torch.tensor(int_values_list, dtype=torch.uint8, device=packed_weights.device)

    # Trim padding
    original_numel = original_shape.numel()
    int_tensor_trimmed = int_tensor[:original_numel]

    unpacked_float_weights = torch.zeros(original_numel, dtype=torch.float32, device=packed_weights.device)

    unpacked_float_weights[int_tensor_trimmed == 0] = 0.0
    unpacked_float_weights[int_tensor_trimmed == 1] = true_positive_value
    unpacked_float_weights[int_tensor_trimmed == 2] = true_negative_value
    # Values of 3 (11_bin) in int_tensor_trimmed would remain 0.0 if not handled,
    # which is acceptable if the packing guarantees only 0, 1, 2 are used.

    return unpacked_float_weights.view(original_shape)