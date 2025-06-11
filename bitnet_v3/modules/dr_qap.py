"""
Dynamic Regularization with Quantization-Aware Penalties (DR-QAP) for BitNet v3.

Implements the adaptive regularization term that stabilizes training:
R_QAP = λ(t) * Σ_i ω_i ||W_i - Q(W_i)||_2^2

where:
- λ(t) decreases as training progresses  
- ω_i weights are computed based on layer sensitivity: ω_i = ||∇_{W_i} L||_2 / Σ_j ||∇_{W_j} L||_2
- Q(W_i) is the quantized version of weights W_i

This provides dynamic regularization that adapts based on quantization error and layer importance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import math
import warnings
from collections import defaultdict
from abc import ABC, abstractmethod

from ..core.quantization import compute_quantization_error, get_layer_sensitivity


class LayerSensitivityCalculator:
    """
    Calculates layer sensitivity weights for dynamic regularization.
    
    Computes ω_i = ||∇_{W_i} L||_2 / Σ_j ||∇_{W_j} L||_2
    """
    
    def __init__(
        self,
        sensitivity_type: str = "gradient_norm",  # "gradient_norm", "fisher", "hessian_trace"
        momentum: float = 0.9,
        min_sensitivity: float = 1e-6,
        normalize_by_layer_size: bool = True,
        use_running_average: bool = True,
    ):
        self.sensitivity_type = sensitivity_type
        self.momentum = momentum
        self.min_sensitivity = min_sensitivity
        self.normalize_by_layer_size = normalize_by_layer_size
        self.use_running_average = use_running_average
        
        # Running averages of sensitivities
        self.running_sensitivities = {}
        self.step_count = 0
    
    def compute_gradient_norm_sensitivity(
        self, 
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute sensitivity based on gradient norms."""
        sensitivities = {}
        total_norm = 0.0
        
        # Compute individual gradient norms
        for layer_name, grad in gradients.items():
            if grad is not None:
                grad_norm = torch.norm(grad, p=2).item()
                
                # Normalize by layer size if requested
                if self.normalize_by_layer_size:
                    grad_norm = grad_norm / math.sqrt(grad.numel())
                
                sensitivities[layer_name] = grad_norm
                total_norm += grad_norm
        
        # Normalize to get relative sensitivities
        if total_norm > self.min_sensitivity:
            for layer_name in sensitivities:
                sensitivities[layer_name] /= total_norm
        else:
            # Uniform sensitivity if total norm is too small
            uniform_weight = 1.0 / len(sensitivities) if sensitivities else 0.0
            for layer_name in sensitivities:
                sensitivities[layer_name] = uniform_weight
        
        return sensitivities
    
    def compute_fisher_sensitivity(
        self,
        model: nn.Module,
        data_loader,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """Compute Fisher Information-based sensitivity (approximate)."""
        # This is a simplified Fisher Information approximation
        model.eval()
        fisher_dict = defaultdict(float)
        
        sample_count = 0
        for batch in data_loader:
            if sample_count >= num_samples:
                break
                
            inputs, targets = batch
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, model.parameters(), 
                retain_graph=False, create_graph=False
            )
            
            # Accumulate squared gradients (Fisher approximation)
            for (name, param), grad in zip(model.named_parameters(), grads):
                if grad is not None:
                    fisher_dict[name] += (grad ** 2).sum().item()
            
            sample_count += inputs.size(0)
        
        # Normalize by total Fisher information
        total_fisher = sum(fisher_dict.values())
        if total_fisher > self.min_sensitivity:
            for name in fisher_dict:
                fisher_dict[name] /= total_fisher
        
        return dict(fisher_dict)
    
    def update_sensitivities(
        self,
        current_sensitivities: Dict[str, float]
    ) -> Dict[str, float]:
        """Update running average of sensitivities."""
        if not self.use_running_average:
            return current_sensitivities
        
        self.step_count += 1
        
        for layer_name, sensitivity in current_sensitivities.items():
            if layer_name not in self.running_sensitivities:
                self.running_sensitivities[layer_name] = sensitivity
            else:
                # Exponential moving average
                self.running_sensitivities[layer_name] = (
                    self.momentum * self.running_sensitivities[layer_name] + 
                    (1 - self.momentum) * sensitivity
                )
        
        return self.running_sensitivities.copy()
    
    def compute_sensitivities(
        self,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        model: Optional[nn.Module] = None,
        data_loader = None,
    ) -> Dict[str, float]:
        """
        Compute layer sensitivities using the specified method.
        
        Args:
            gradients: Dictionary of layer gradients
            model: Model for Fisher information calculation
            data_loader: Data loader for Fisher information
            
        Returns:
            Dictionary of layer sensitivity weights
        """
        if self.sensitivity_type == "gradient_norm":
            if gradients is None:
                raise ValueError("Gradients required for gradient_norm sensitivity")
            current_sensitivities = self.compute_gradient_norm_sensitivity(gradients)
            
        elif self.sensitivity_type == "fisher":
            if model is None or data_loader is None:
                raise ValueError("Model and data_loader required for Fisher sensitivity")
            current_sensitivities = self.compute_fisher_sensitivity(model, data_loader)
            
        else:
            raise ValueError(f"Unknown sensitivity type: {self.sensitivity_type}")
        
        # Update running averages
        return self.update_sensitivities(current_sensitivities)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sensitivity computation statistics."""
        return {
            'step_count': self.step_count,
            'num_layers': len(self.running_sensitivities),
            'sensitivity_values': self.running_sensitivities.copy(),
            'sensitivity_stats': {
                'min': min(self.running_sensitivities.values()) if self.running_sensitivities else 0.0,
                'max': max(self.running_sensitivities.values()) if self.running_sensitivities else 0.0,
                'mean': sum(self.running_sensitivities.values()) / len(self.running_sensitivities) if self.running_sensitivities else 0.0,
            }
        }


class LambdaScheduler(ABC):
    """Base class for λ(t) scheduling in DR-QAP."""
    
    @abstractmethod
    def get_lambda(self, step: int, epoch: int, total_epochs: int) -> float:
        """Get regularization strength for current step."""
        pass


class ExponentialDecayScheduler(LambdaScheduler):
    """Exponential decay: λ(t) = λ_0 * decay_factor^t"""
    
    def __init__(self, initial_lambda: float = 0.1, decay_factor: float = 0.95):
        self.initial_lambda = initial_lambda
        self.decay_factor = decay_factor
    
    def get_lambda(self, step: int, epoch: int, total_epochs: int) -> float:
        return self.initial_lambda * (self.decay_factor ** epoch)


class CosineDecayScheduler(LambdaScheduler):
    """Cosine decay schedule for smooth transitions."""
    
    def __init__(self, initial_lambda: float = 0.1, final_lambda: float = 0.01):
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
    
    def get_lambda(self, step: int, epoch: int, total_epochs: int) -> float:
        progress = epoch / max(1, total_epochs)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_lambda + (self.initial_lambda - self.final_lambda) * cosine_factor


class StepDecayScheduler(LambdaScheduler):
    """Step-wise decay at specific epochs."""
    
    def __init__(
        self, 
        initial_lambda: float = 0.1, 
        decay_factor: float = 0.5,
        decay_epochs: List[int] = None
    ):
        self.initial_lambda = initial_lambda
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs or [20, 40, 60]
    
    def get_lambda(self, step: int, epoch: int, total_epochs: int) -> float:
        decay_count = sum(1 for e in self.decay_epochs if epoch >= e)
        return self.initial_lambda * (self.decay_factor ** decay_count)


class QuantizationAwarePenalty(nn.Module):
    """
    Computes the quantization-aware penalty term:
    R_QAP = λ(t) * Σ_i ω_i ||W_i - Q(W_i)||_2^2
    """
    
    def __init__(
        self,
        penalty_type: str = "l2",  # "l2", "l1", "huber"
        huber_delta: float = 1.0,
        normalize_by_layer_size: bool = True,
        accumulate_over_layers: bool = True,
    ):
        super().__init__()
        self.penalty_type = penalty_type
        self.huber_delta = huber_delta
        self.normalize_by_layer_size = normalize_by_layer_size
        self.accumulate_over_layers = accumulate_over_layers
    
    def forward(
        self,
        original_weights: Dict[str, torch.Tensor],
        quantized_weights: Dict[str, torch.Tensor],
        layer_weights: Dict[str, float],
        lambda_value: float,
    ) -> torch.Tensor:
        """
        Compute quantization-aware penalty.
        
        Args:
            original_weights: Dictionary of original weight tensors
            quantized_weights: Dictionary of quantized weight tensors
            layer_weights: Dictionary of layer sensitivity weights (ω_i)
            lambda_value: Current regularization strength λ(t)
            
        Returns:
            Penalty term tensor
        """
        total_penalty = 0.0
        layer_count = 0
        
        for layer_name in original_weights:
            if layer_name not in quantized_weights:
                warnings.warn(f"Layer {layer_name} not found in quantized weights")
                continue
            
            if layer_name not in layer_weights:
                warnings.warn(f"Layer {layer_name} not found in layer weights")
                continue
            
            orig_weight = original_weights[layer_name]
            quant_weight = quantized_weights[layer_name]
            layer_weight = layer_weights[layer_name]
            
            # Ensure same shape
            if orig_weight.shape != quant_weight.shape:
                warnings.warn(f"Shape mismatch for layer {layer_name}")
                continue
            
            # Compute quantization error
            error = orig_weight - quant_weight
            
            # Compute penalty based on type
            if self.penalty_type == "l2":
                layer_penalty = torch.norm(error, p=2) ** 2
            elif self.penalty_type == "l1":
                layer_penalty = torch.norm(error, p=1)
            elif self.penalty_type == "huber":
                # Huber loss: smooth combination of L1 and L2
                abs_error = torch.abs(error)
                huber_mask = abs_error <= self.huber_delta
                
                l2_loss = 0.5 * (error ** 2)
                l1_loss = self.huber_delta * abs_error - 0.5 * (self.huber_delta ** 2)
                
                layer_penalty = torch.where(huber_mask, l2_loss, l1_loss).sum()
            else:
                raise ValueError(f"Unknown penalty type: {self.penalty_type}")
            
            # Normalize by layer size if requested
            if self.normalize_by_layer_size:
                layer_penalty = layer_penalty / orig_weight.numel()
            
            # Apply layer weight
            weighted_penalty = layer_weight * layer_penalty
            
            if self.accumulate_over_layers:
                total_penalty += weighted_penalty
            else:
                # Return individual layer penalties
                total_penalty = weighted_penalty if layer_count == 0 else torch.stack([total_penalty, weighted_penalty])
            
            layer_count += 1
        
        # Apply global lambda scaling
        return lambda_value * total_penalty


class DynamicRegularizationQAP(nn.Module):
    """
    Main Dynamic Regularization with Quantization-Aware Penalties module.
    
    Integrates all components: sensitivity calculation, lambda scheduling, and penalty computation.
    """
    
    def __init__(
        self,
        initial_lambda: float = 0.1,
        lambda_scheduler: Optional[LambdaScheduler] = None,
        sensitivity_calculator: Optional[LayerSensitivityCalculator] = None,
        penalty_module: Optional[QuantizationAwarePenalty] = None,
        update_frequency: int = 1,  # Update sensitivities every N steps
        warmup_steps: int = 100,    # Steps before applying regularization
        quantize_function: Optional[Callable] = None,
    ):
        super().__init__()
        
        # Initialize components
        self.lambda_scheduler = lambda_scheduler or ExponentialDecayScheduler(initial_lambda)
        self.sensitivity_calculator = sensitivity_calculator or LayerSensitivityCalculator()
        self.penalty_module = penalty_module or QuantizationAwarePenalty()
        
        self.update_frequency = update_frequency
        self.warmup_steps = warmup_steps
        self.quantize_function = quantize_function
        
        # State tracking
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('epoch_count', torch.tensor(0, dtype=torch.long))
        self.total_epochs = 70  # Default, can be updated
        
        # Statistics
        self.penalty_history = []
        self.lambda_history = []
        self.sensitivity_history = []
        
        # Cache for efficiency
        self.cached_sensitivities = {}
        self.cached_lambda = 0.0
    
    def set_total_epochs(self, total_epochs: int):
        """Set total training epochs for scheduling."""
        self.total_epochs = total_epochs
    
    def set_quantize_function(self, quantize_fn: Callable):
        """Set the quantization function to use."""
        self.quantize_function = quantize_fn
    
    def update_epoch(self, epoch: int):
        """Update current epoch for scheduling."""
        self.epoch_count = torch.tensor(epoch, dtype=torch.long)
    
    def _should_update_sensitivities(self) -> bool:
        """Check if sensitivities should be updated this step."""
        return (self.step_count % self.update_frequency == 0 and 
                self.step_count >= self.warmup_steps)
    
    def _get_current_lambda(self) -> float:
        """Get current regularization strength."""
        if self.step_count < self.warmup_steps:
            return 0.0  # No regularization during warmup
        
        return self.lambda_scheduler.get_lambda(
            self.step_count.item(), 
            self.epoch_count.item(), 
            self.total_epochs
        )
    
    def forward(
        self,
        model: nn.Module,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        target_layers: Optional[List[str]] = None,
        data_loader = None,
    ) -> torch.Tensor:
        """
        Compute dynamic regularization penalty.
        
        Args:
            model: The model being trained
            gradients: Dictionary of layer gradients (for sensitivity)
            target_layers: Specific layers to regularize (if None, use all)
            data_loader: Data loader for Fisher information (if needed)
            
        Returns:
            Regularization penalty tensor
        """
        self.step_count += 1
        
        # Get current lambda
        current_lambda = self._get_current_lambda()
        self.cached_lambda = current_lambda
        self.lambda_history.append(current_lambda)
        
        # Early exit if no regularization
        if current_lambda == 0.0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Update sensitivities if needed
        if self._should_update_sensitivities():
            sensitivities = self.sensitivity_calculator.compute_sensitivities(
                gradients=gradients,
                model=model,
                data_loader=data_loader,
            )
            self.cached_sensitivities = sensitivities
            self.sensitivity_history.append(sensitivities)
        
        # Use cached sensitivities if no update
        if not self.cached_sensitivities:
            # Uniform weights as fallback
            param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
            if target_layers:
                param_names = [name for name in param_names if any(tl in name for tl in target_layers)]
            
            uniform_weight = 1.0 / len(param_names) if param_names else 0.0
            self.cached_sensitivities = {name: uniform_weight for name in param_names}
        
        # Collect original and quantized weights
        original_weights = {}
        quantized_weights = {}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if target_layers and not any(tl in name for tl in target_layers):
                continue
            
            original_weights[name] = param.data
            
            # Apply quantization
            if self.quantize_function is not None:
                quantized_weights[name] = self.quantize_function(param.data)
            else:
                # Default quantization (placeholder)
                quantized_weights[name] = param.data.round()
        
        # Compute penalty
        if original_weights and quantized_weights:
            penalty = self.penalty_module(
                original_weights=original_weights,
                quantized_weights=quantized_weights,
                layer_weights=self.cached_sensitivities,
                lambda_value=current_lambda,
            )
            
            self.penalty_history.append(penalty.item())
        else:
            penalty = torch.tensor(0.0, device=next(model.parameters()).device)
            self.penalty_history.append(0.0)
        
        return penalty
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'step_count': self.step_count.item(),
            'epoch_count': self.epoch_count.item(),
            'current_lambda': self.cached_lambda,
            'current_sensitivities': self.cached_sensitivities.copy(),
            'penalty_history': self.penalty_history[-100:],  # Last 100 steps
            'lambda_history': self.lambda_history[-100:],
            'sensitivity_stats': self.sensitivity_calculator.get_statistics(),
        }
        
        # Add summary statistics
        if self.penalty_history:
            recent_penalties = self.penalty_history[-10:]
            stats['penalty_stats'] = {
                'recent_mean': sum(recent_penalties) / len(recent_penalties),
                'recent_max': max(recent_penalties),
                'recent_min': min(recent_penalties),
            }
        
        return stats
    
    def get_regularization_strength(self) -> float:
        """Get current regularization strength."""
        return self.cached_lambda
    
    def get_layer_sensitivities(self) -> Dict[str, float]:
        """Get current layer sensitivity weights."""
        return self.cached_sensitivities.copy()
    
    def extra_repr(self) -> str:
        return (f'step_count={self.step_count.item()}, '
                f'lambda={self.cached_lambda:.6f}, '
                f'num_sensitive_layers={len(self.cached_sensitivities)}, '
                f'warmup_steps={self.warmup_steps}')