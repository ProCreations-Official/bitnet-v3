"""
Enhanced Straight-Through Estimator with Momentum (ESTE-M) for BitNet v3.

Implements the enhanced gradient estimator that improves gradient approximation:
g_t = μ · g_{t-1} + (1-μ) · ∇_W L
∇̃_W = sign(g_t) · |g_t|^{0.5}

This provides more stable gradient signals, especially during the transition 
between quantization levels, with variance reduced by 43% compared to standard STE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Union, Callable
import math
import warnings
from collections import defaultdict

from ..core.straight_through import (
    EnhancedSTEWithMomentum as CoreEnhancedSTE,
    MomentumGradientBuffer,
    reset_all_momentum_buffers,
)


class MomentumGradientEstimator(nn.Module):
    """
    Enhanced gradient estimator with momentum for stable quantization training.
    
    Implements the ESTE-M algorithm from BitNet v3 paper.
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        power: float = 0.5,
        use_adaptive_momentum: bool = True,
        momentum_warmup_steps: int = 100,
        gradient_clipping: Optional[float] = None,
        variance_tracking: bool = True,
        buffer_id_prefix: str = "este_m",
    ):
        super().__init__()
        
        self.momentum = momentum
        self.power = power
        self.use_adaptive_momentum = use_adaptive_momentum
        self.momentum_warmup_steps = momentum_warmup_steps
        self.gradient_clipping = gradient_clipping
        self.variance_tracking = variance_tracking
        self.buffer_id_prefix = buffer_id_prefix
        
        # Step tracking
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        
        # Statistics tracking
        if variance_tracking:
            self.register_buffer('gradient_variance_raw', torch.tensor(0.0))
            self.register_buffer('gradient_variance_enhanced', torch.tensor(0.0))
            self.register_buffer('variance_reduction_ratio', torch.tensor(0.0))
        
        # Per-layer momentum buffers
        self.layer_buffers = {}
        
        # Adaptive momentum parameters
        if use_adaptive_momentum:
            self.register_buffer('recent_gradient_norms', torch.zeros(10))
            self.norm_history_idx = 0
    
    def _get_adaptive_momentum(self, gradient_norm: float) -> float:
        """
        Compute adaptive momentum based on recent gradient behavior.
        
        Higher momentum when gradients are stable, lower when volatile.
        """
        if not self.use_adaptive_momentum:
            return self.momentum
        
        # Update gradient norm history
        self.recent_gradient_norms[self.norm_history_idx % 10] = gradient_norm
        self.norm_history_idx += 1
        
        if self.norm_history_idx < 10:
            return self.momentum  # Not enough history
        
        # Compute gradient stability (inverse of coefficient of variation)
        recent_norms = self.recent_gradient_norms
        mean_norm = recent_norms.mean()
        std_norm = recent_norms.std()
        
        if mean_norm > 1e-8:
            stability = 1.0 / (1.0 + std_norm / mean_norm)
        else:
            stability = 0.5
        
        # Adaptive momentum: higher stability -> higher momentum
        min_momentum = 0.5
        max_momentum = 0.99
        adaptive_momentum = min_momentum + stability * (max_momentum - min_momentum)
        
        return float(adaptive_momentum)
    
    def _get_layer_buffer_id(self, layer_name: str) -> str:
        """Generate unique buffer ID for layer."""
        return f"{self.buffer_id_prefix}_{layer_name}_{id(self)}"
    
    def _apply_momentum_smoothing(
        self, 
        gradient: torch.Tensor, 
        layer_name: str,
        momentum: float
    ) -> torch.Tensor:
        """
        Apply momentum smoothing to gradient.
        
        g_t = μ · g_{t-1} + (1-μ) · ∇_W L
        """
        buffer_id = self._get_layer_buffer_id(layer_name)
        
        if buffer_id not in self.layer_buffers:
            self.layer_buffers[buffer_id] = MomentumGradientBuffer(momentum)
        
        # Update momentum if adaptive
        self.layer_buffers[buffer_id].momentum = momentum
        
        return self.layer_buffers[buffer_id].update(gradient)
    
    def _apply_power_transformation(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Apply power transformation: sign(g_t) · |g_t|^{power}
        """
        sign = gradient.sign()
        magnitude = gradient.abs().pow(self.power)
        return sign * magnitude
    
    def _clip_gradients(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply gradient clipping if configured."""
        if self.gradient_clipping is not None:
            gradient = gradient.clamp(-self.gradient_clipping, self.gradient_clipping)
        return gradient
    
    def _update_variance_statistics(
        self, 
        raw_gradient: torch.Tensor, 
        enhanced_gradient: torch.Tensor
    ):
        """Update gradient variance tracking statistics."""
        if not self.variance_tracking:
            return
        
        with torch.no_grad():
            # Compute variances
            raw_var = raw_gradient.var().item()
            enhanced_var = enhanced_gradient.var().item()
            
            # Exponential moving average
            momentum = 0.99
            self.gradient_variance_raw = momentum * self.gradient_variance_raw + (1 - momentum) * raw_var
            self.gradient_variance_enhanced = momentum * self.gradient_variance_enhanced + (1 - momentum) * enhanced_var
            
            # Compute variance reduction ratio
            if self.gradient_variance_raw > 1e-8:
                self.variance_reduction_ratio = self.gradient_variance_enhanced / self.gradient_variance_raw
    
    def enhance_gradient(
        self, 
        gradient: torch.Tensor, 
        layer_name: str = "default"
    ) -> torch.Tensor:
        """
        Apply ESTE-M enhancement to a single gradient tensor.
        
        Args:
            gradient: Raw gradient tensor
            layer_name: Identifier for the layer (for separate momentum buffers)
            
        Returns:
            Enhanced gradient with momentum smoothing and power transformation
        """
        self.step_count += 1
        
        # Apply clipping first
        clipped_gradient = self._clip_gradients(gradient)
        
        # Compute adaptive momentum
        gradient_norm = torch.norm(clipped_gradient).item()
        current_momentum = self._get_adaptive_momentum(gradient_norm)
        
        # Apply warmup to momentum
        if self.step_count < self.momentum_warmup_steps:
            warmup_factor = self.step_count / self.momentum_warmup_steps
            current_momentum = warmup_factor * current_momentum
        
        # Apply momentum smoothing
        smoothed_gradient = self._apply_momentum_smoothing(
            clipped_gradient, layer_name, current_momentum
        )
        
        # Apply power transformation
        enhanced_gradient = self._apply_power_transformation(smoothed_gradient)
        
        # Update statistics
        self._update_variance_statistics(clipped_gradient, enhanced_gradient)
        
        return enhanced_gradient
    
    def enhance_model_gradients(
        self, 
        model: nn.Module,
        target_layers: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply ESTE-M to all model gradients.
        
        Args:
            model: PyTorch model
            target_layers: List of layer names to enhance (if None, enhance all)
            
        Returns:
            Dictionary of enhanced gradients
        """
        enhanced_gradients = {}
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
                
            if target_layers is not None and name not in target_layers:
                continue
            
            enhanced_grad = self.enhance_gradient(param.grad.data, name)
            enhanced_gradients[name] = enhanced_grad
            
            # Update the actual gradient
            param.grad.data.copy_(enhanced_grad)
        
        return enhanced_gradients
    
    def reset_momentum_buffers(self):
        """Reset all momentum buffers."""
        for buffer in self.layer_buffers.values():
            buffer.reset()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics and performance metrics."""
        stats = {
            'step_count': self.step_count.item(),
            'num_layer_buffers': len(self.layer_buffers),
            'current_momentum': self.momentum,
            'power': self.power,
        }
        
        if self.variance_tracking:
            stats.update({
                'gradient_variance_raw': self.gradient_variance_raw.item(),
                'gradient_variance_enhanced': self.gradient_variance_enhanced.item(),
                'variance_reduction_ratio': self.variance_reduction_ratio.item(),
                'variance_reduction_percent': (1 - self.variance_reduction_ratio.item()) * 100,
            })
        
        if self.use_adaptive_momentum:
            stats.update({
                'recent_gradient_norms': self.recent_gradient_norms.tolist(),
                'adaptive_momentum': self._get_adaptive_momentum(
                    self.recent_gradient_norms.mean().item()
                ),
            })
        
        return stats
    
    def extra_repr(self) -> str:
        return (f'momentum={self.momentum}, power={self.power}, '
                f'adaptive={self.use_adaptive_momentum}, '
                f'variance_tracking={self.variance_tracking}')


class EnhancedSTEMomentum(nn.Module):
    """
    Integrated ESTE-M module for use in quantized layers.
    
    Combines the core STE functionality with momentum-based gradient enhancement.
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        power: float = 0.5,
        clip_value: Optional[float] = None,
        enable_enhanced_ste: bool = True,
        gradient_estimator_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.momentum = momentum
        self.power = power
        self.clip_value = clip_value
        self.enable_enhanced_ste = enable_enhanced_ste
        
        # Initialize gradient estimator
        estimator_config = gradient_estimator_config or {}
        estimator_config.update({
            'momentum': momentum,
            'power': power,
            'gradient_clipping': clip_value,
        })
        
        self.gradient_estimator = MomentumGradientEstimator(**estimator_config)
        
        # Layer-specific buffer ID for the core STE
        self.buffer_id = f"enhanced_ste_{id(self)}"
    
    def forward(
        self, 
        input_tensor: torch.Tensor, 
        quantize_function: Callable,
        layer_name: str = "default",
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Apply quantization with enhanced STE gradient estimation.
        
        Args:
            input_tensor: Input tensor to quantize
            quantize_function: Function to apply quantization
            layer_name: Name of the layer (for gradient tracking)
            *args, **kwargs: Additional arguments for quantization function
            
        Returns:
            Quantized tensor with enhanced gradient flow
        """
        if self.enable_enhanced_ste:
            # Use the enhanced STE with momentum from core module
            return CoreEnhancedSTE.apply(
                input_tensor, 
                quantize_function, 
                self.momentum, 
                self.power, 
                f"{self.buffer_id}_{layer_name}",
                *args, **kwargs
            )
        else:
            # Use standard STE
            class StandardSTE(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, qfn, *args, **kwargs):
                    return qfn(x, *args, **kwargs)
                
                @staticmethod
                def backward(ctx, grad_output):
                    return grad_output, None, None, None
            
            return StandardSTE.apply(input_tensor, quantize_function, *args, **kwargs)
    
    def post_backward_hook(self, model: nn.Module, layer_names: Optional[list] = None):
        """
        Apply gradient enhancement after backward pass.
        
        This should be called after loss.backward() but before optimizer.step().
        """
        if self.enable_enhanced_ste:
            return self.gradient_estimator.enhance_model_gradients(model, layer_names)
        return {}
    
    def reset_state(self):
        """Reset all internal state."""
        self.gradient_estimator.reset_momentum_buffers()
        # Also reset core STE buffers
        reset_all_momentum_buffers()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics including variance reduction."""
        return self.gradient_estimator.get_statistics()
    
    def adapt_parameters(self, **kwargs):
        """Dynamically adapt ESTE-M parameters."""
        if 'momentum' in kwargs:
            self.momentum = kwargs['momentum']
            self.gradient_estimator.momentum = kwargs['momentum']
        
        if 'power' in kwargs:
            self.power = kwargs['power'] 
            self.gradient_estimator.power = kwargs['power']
        
        if 'clip_value' in kwargs:
            self.clip_value = kwargs['clip_value']
            self.gradient_estimator.gradient_clipping = kwargs['clip_value']
    
    def extra_repr(self) -> str:
        return (f'momentum={self.momentum}, power={self.power}, '
                f'clip_value={self.clip_value}, enhanced_ste={self.enable_enhanced_ste}')


class ESTEMTrainingCallback:
    """
    Training callback for ESTE-M integration and monitoring.
    """
    
    def __init__(
        self, 
        este_m_modules: Dict[str, EnhancedSTEMomentum],
        log_frequency: int = 100,
        adapt_parameters: bool = False,
    ):
        self.este_m_modules = este_m_modules
        self.log_frequency = log_frequency
        self.adapt_parameters = adapt_parameters
        
        self.step_count = 0
        self.performance_history = defaultdict(list)
    
    def on_backward_end(self, model: nn.Module):
        """Called after backward pass, before optimizer step."""
        self.step_count += 1
        
        # Apply gradient enhancement for all ESTE-M modules
        for module_name, este_m in self.este_m_modules.items():
            este_m.post_backward_hook(model)
    
    def on_step_end(self, optimizer, **kwargs):
        """Called after optimizer step."""
        # Log performance metrics
        if self.step_count % self.log_frequency == 0:
            self._log_performance_metrics()
        
        # Adaptive parameter adjustment
        if self.adapt_parameters:
            self._adapt_parameters(**kwargs)
    
    def _log_performance_metrics(self):
        """Log performance metrics from all ESTE-M modules."""
        for module_name, este_m in self.este_m_modules.items():
            metrics = este_m.get_performance_metrics()
            
            for metric_name, value in metrics.items():
                key = f"{module_name}_{metric_name}"
                self.performance_history[key].append(value)
    
    def _adapt_parameters(self, **kwargs):
        """Adapt ESTE-M parameters based on training progress."""
        # Simple adaptation example: reduce momentum over time
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']
            total_epochs = kwargs.get('total_epochs', 100)
            
            # Gradually reduce momentum
            progress = epoch / total_epochs
            adapted_momentum = 0.9 * (1 - 0.3 * progress)  # Reduce by up to 30%
            
            for este_m in self.este_m_modules.values():
                este_m.adapt_parameters(momentum=adapted_momentum)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'total_steps': self.step_count,
            'num_modules': len(self.este_m_modules),
            'performance_history': dict(self.performance_history),
        }
        
        # Compute average variance reduction
        variance_reductions = []
        for module_name, este_m in self.este_m_modules.items():
            metrics = este_m.get_performance_metrics()
            if 'variance_reduction_percent' in metrics:
                variance_reductions.append(metrics['variance_reduction_percent'])
        
        if variance_reductions:
            summary['average_variance_reduction'] = sum(variance_reductions) / len(variance_reductions)
        
        return summary
    
    def reset_all_states(self):
        """Reset all ESTE-M module states."""
        for este_m in self.este_m_modules.values():
            este_m.reset_state()


# Utility functions for easy integration
def create_este_m_module(
    momentum: float = 0.9,
    power: float = 0.5,
    **kwargs
) -> EnhancedSTEMomentum:
    """Create an ESTE-M module with default settings."""
    return EnhancedSTEMomentum(momentum=momentum, power=power, **kwargs)


def integrate_este_m_with_model(
    model: nn.Module,
    target_layer_types: tuple = (nn.Linear,),
    **este_m_kwargs
) -> Dict[str, EnhancedSTEMomentum]:
    """
    Integrate ESTE-M modules with model layers.
    
    Args:
        model: PyTorch model
        target_layer_types: Layer types to integrate with
        **este_m_kwargs: Arguments for ESTE-M modules
        
    Returns:
        Dictionary of layer_name -> EnhancedSTEMomentum mappings
    """
    este_m_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, target_layer_types):
            este_m = create_este_m_module(**este_m_kwargs)
            este_m_modules[name] = este_m
            
            # Could attach as attribute for automatic integration
            setattr(module, '_este_m', este_m)
    
    return este_m_modules