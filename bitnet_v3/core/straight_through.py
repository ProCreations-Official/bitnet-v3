"""
Straight-Through Estimator implementations for BitNet v3.

Implements various gradient estimators for quantized neural networks:
- Standard Straight-Through Estimator (STE)
- Enhanced STE with Momentum (ESTE-M) from BitNet v3
- Gradient clipping and smoothing utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class StraightThroughEstimator(torch.autograd.Function):
    """
    Standard Straight-Through Estimator.
    
    Forward: Apply quantization function
    Backward: Pass gradients through unchanged
    """
    
    @staticmethod
    def forward(ctx, input, quantize_fn, *args, **kwargs):
        """
        Forward pass with quantization.
        
        Args:
            ctx: Context for backward pass
            input: Input tensor
            quantize_fn: Quantization function to apply
            *args, **kwargs: Arguments for quantization function
            
        Returns:
            Quantized tensor
        """
        ctx.save_for_backward(input)
        return quantize_fn(input, *args, **kwargs)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with straight-through gradients.
        
        Args:
            ctx: Context from forward pass
            grad_output: Gradients from subsequent layers
            
        Returns:
            Gradients for input (passed through unchanged)
        """
        return grad_output, None, None


class ClippedStraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator with gradient clipping.
    """
    
    @staticmethod
    def forward(ctx, input, quantize_fn, clip_value=1.0, *args, **kwargs):
        ctx.save_for_backward(input)
        ctx.clip_value = clip_value
        return quantize_fn(input, *args, **kwargs)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        clip_value = ctx.clip_value
        
        # Clip gradients to range [-clip_value, clip_value]
        grad_input = grad_output.clamp(-clip_value, clip_value)
        
        return grad_input, None, None


class MomentumGradientBuffer:
    """
    Momentum buffer for gradient smoothing in ESTE-M.
    """
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.buffer: Optional[torch.Tensor] = None
    
    def update(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Update momentum buffer and return smoothed gradient.
        
        Args:
            gradient: Current gradient tensor
            
        Returns:
            Momentum-smoothed gradient
        """
        if self.buffer is None:
            self.buffer = gradient.clone().detach()
            return gradient
        
        # Apply momentum: g_t = μ * g_{t-1} + (1-μ) * ∇L
        self.buffer = self.momentum * self.buffer + (1 - self.momentum) * gradient
        return self.buffer
    
    def reset(self):
        """Reset the momentum buffer."""
        self.buffer = None


class EnhancedSTEWithMomentum(torch.autograd.Function):
    """
    Enhanced Straight-Through Estimator with Momentum (ESTE-M) from BitNet v3.
    
    Implements the gradient smoothing equation:
    g_t = μ * g_{t-1} + (1-μ) * ∇W L
    ∇̃W = sign(g_t) * |g_t|^{0.5}
    """
    
    momentum_buffers: Dict[str, MomentumGradientBuffer] = {}
    
    @staticmethod
    def forward(ctx, input, quantize_fn, momentum=0.9, power=0.5, buffer_id="default", *args, **kwargs):
        ctx.save_for_backward(input)
        ctx.momentum = momentum
        ctx.power = power
        ctx.buffer_id = buffer_id
        
        return quantize_fn(input, *args, **kwargs)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        momentum = ctx.momentum
        power = ctx.power
        buffer_id = ctx.buffer_id
        
        # Get or create momentum buffer
        if buffer_id not in EnhancedSTEWithMomentum.momentum_buffers:
            EnhancedSTEWithMomentum.momentum_buffers[buffer_id] = MomentumGradientBuffer(momentum)
        
        buffer = EnhancedSTEWithMomentum.momentum_buffers[buffer_id]
        
        # Apply momentum smoothing
        smoothed_grad = buffer.update(grad_output)
        
        # Apply power transformation: sign(g_t) * |g_t|^{power}
        sign = smoothed_grad.sign()
        magnitude = smoothed_grad.abs().pow(power)
        enhanced_grad = sign * magnitude
        
        return enhanced_grad, None, None, None, None, None
    
    @staticmethod
    def reset_buffers():
        """Reset all momentum buffers."""
        for buffer in EnhancedSTEWithMomentum.momentum_buffers.values():
            buffer.reset()


class EnhancedSTEMomentum(nn.Module):
    """
    Enhanced STE with Momentum module for easy integration.
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        power: float = 0.5,
        clip_value: Optional[float] = None,
        buffer_id: Optional[str] = None
    ):
        super().__init__()
        self.momentum = momentum
        self.power = power
        self.clip_value = clip_value
        self.buffer_id = buffer_id or f"buffer_{id(self)}"
        
        # Internal momentum buffer
        self.momentum_buffer = MomentumGradientBuffer(momentum)
    
    def forward(self, input: torch.Tensor, quantize_fn, *args, **kwargs) -> torch.Tensor:
        """
        Apply quantization with enhanced STE gradient estimation.
        
        Args:
            input: Input tensor
            quantize_fn: Quantization function
            *args, **kwargs: Arguments for quantization function
            
        Returns:
            Quantized tensor with enhanced gradient flow
        """
        if self.clip_value is not None:
            return ClippedStraightThroughEstimator.apply(
                input, quantize_fn, self.clip_value, *args, **kwargs
            )
        else:
            return EnhancedSTEWithMomentum.apply(
                input, quantize_fn, self.momentum, self.power, self.buffer_id, *args, **kwargs
            )
    
    def reset_momentum(self):
        """Reset momentum buffer."""
        self.momentum_buffer.reset()
    
    def extra_repr(self) -> str:
        return f'momentum={self.momentum}, power={self.power}, clip_value={self.clip_value}'


class GradientAnalyzer:
    """
    Utility class for analyzing gradient flow in quantized networks.
    """
    
    def __init__(self):
        self.gradient_stats = []
    
    def analyze_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze gradient statistics for a model.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary of gradient statistics
        """
        stats = {
            'total_params': 0,
            'total_grad_norm': 0.0,
            'layer_stats': {},
            'gradient_variance': 0.0,
            'dead_neurons': 0,
        }
        
        total_params = 0
        total_grad_norm_sq = 0.0
        all_gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                
                # Count dead neurons (gradients very close to zero)
                dead_count = (grad.abs() < 1e-8).sum().item()
                
                stats['layer_stats'][name] = {
                    'grad_norm': grad_norm,
                    'grad_mean': grad_mean,
                    'grad_std': grad_std,
                    'shape': tuple(grad.shape),
                    'dead_neurons': dead_count,
                }
                
                total_params += param.numel()
                total_grad_norm_sq += grad_norm ** 2
                all_gradients.append(grad.view(-1))
                stats['dead_neurons'] += dead_count
        
        if all_gradients:
            all_grads = torch.cat(all_gradients)
            stats['gradient_variance'] = all_grads.var().item()
        
        stats['total_params'] = total_params
        stats['total_grad_norm'] = math.sqrt(total_grad_norm_sq)
        
        return stats
    
    def log_gradient_stats(self, model: nn.Module, step: int):
        """Log gradient statistics for monitoring."""
        stats = self.analyze_gradients(model)
        self.gradient_stats.append({
            'step': step,
            'total_grad_norm': stats['total_grad_norm'],
            'gradient_variance': stats['gradient_variance'],
            'dead_neurons': stats['dead_neurons'],
        })


class QuantizationAwareGradientClipper:
    """
    Gradient clipper that adapts to quantization levels.
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        quantization_aware: bool = True
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.quantization_aware = quantization_aware
    
    def clip_gradients(
        self,
        parameters,
        current_quantization_level: Optional[float] = None
    ) -> float:
        """
        Clip gradients with optional quantization-aware scaling.
        
        Args:
            parameters: Model parameters
            current_quantization_level: Current quantization bit-width
            
        Returns:
            Total gradient norm before clipping
        """
        # Compute total gradient norm
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), self.norm_type)
                for p in parameters if p.grad is not None
            ]),
            self.norm_type
        )
        
        # Adapt clipping threshold based on quantization level
        if self.quantization_aware and current_quantization_level is not None:
            # Lower precision -> more aggressive clipping
            scale_factor = max(0.1, current_quantization_level / 16.0)
            adapted_max_norm = self.max_norm * scale_factor
        else:
            adapted_max_norm = self.max_norm
        
        # Apply clipping
        clip_coef = adapted_max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.detach().mul_(clip_coef)
        
        return total_norm.item()


def apply_straight_through_estimator(
    input_tensor: torch.Tensor,
    quantize_function,
    estimator_type: str = "standard",
    **kwargs
) -> torch.Tensor:
    """
    Convenience function to apply different STE variants.
    
    Args:
        input_tensor: Input tensor to quantize
        quantize_function: Quantization function to apply
        estimator_type: Type of STE ("standard", "clipped", "enhanced")
        **kwargs: Additional arguments for the estimator
        
    Returns:
        Quantized tensor with appropriate gradient flow
    """
    if estimator_type == "standard":
        return StraightThroughEstimator.apply(input_tensor, quantize_function)
    elif estimator_type == "clipped":
        clip_value = kwargs.get("clip_value", 1.0)
        return ClippedStraightThroughEstimator.apply(input_tensor, quantize_function, clip_value)
    elif estimator_type == "enhanced":
        momentum = kwargs.get("momentum", 0.9)
        power = kwargs.get("power", 0.5)
        buffer_id = kwargs.get("buffer_id", "default")
        return EnhancedSTEWithMomentum.apply(
            input_tensor, quantize_function, momentum, power, buffer_id
        )
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


# Utility function for resetting all momentum buffers
def reset_all_momentum_buffers():
    """Reset all global momentum buffers."""
    EnhancedSTEWithMomentum.reset_buffers()