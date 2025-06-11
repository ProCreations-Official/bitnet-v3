"""
Enhanced H-BitLinear module for BitNet v3.

Integrates all five key innovations into a single linear layer:
1. Multi-stage Progressive Quantization (MPQ)
2. Adaptive Hadamard Transform with Learnable Parameters (AHT-LP)
3. Gradient-Aware Knowledge Distillation (GAKD) - integrated via training
4. Dynamic Regularization with Quantization-Aware Penalties (DR-QAP) - integrated via training
5. Enhanced Straight-Through Estimator with Momentum (ESTE-M)

This replaces standard nn.Linear layers in transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import warnings
from dataclasses import dataclass

from ..core.quantization import (
    quantize_weights_158, 
    quantize_activations,
    progressive_quantize,
    QuantizationConfig
)
from ..core.hadamard import fast_hadamard_transform, is_power_of_2
from .aht_lp import AdaptiveHadamardTransform
from .mpq import MultiStageProgressiveQuantizer, MPQConfig
from .este_m import EnhancedSTEMomentum


@dataclass
class BitLinearConfig:
    """Configuration for Enhanced H-BitLinear layer."""
    
    # Basic layer configuration
    in_features: int
    out_features: int
    bias: bool = False
    
    # Quantization configuration
    weight_bits: float = 1.58
    activation_bits: int = 4
    weight_quantization_method: str = "absmean"
    activation_quantization_method: str = "absmax"
    
    # MPQ configuration
    enable_mpq: bool = True
    mpq_config: Optional[MPQConfig] = None
    
    # AHT-LP configuration
    enable_adaptive_hadamard: bool = True
    hadamard_normalize: bool = True
    hadamard_learnable_scale: bool = True
    hadamard_learnable_shift: bool = True
    hadamard_init_scale: float = 1.0
    hadamard_init_shift: float = 0.0
    hadamard_scale_lr_multiplier: float = 1.0
    hadamard_shift_lr_multiplier: float = 1.0
    
    # ESTE-M configuration
    enable_enhanced_ste: bool = True
    ste_momentum: float = 0.9
    ste_power: float = 0.5
    ste_clip_value: Optional[float] = None
    
    # Performance optimizations
    use_fast_hadamard: bool = True
    enable_statistics: bool = True
    pad_to_power_of_2: bool = True


class EnhancedHBitLinear(nn.Module):
    """
    Enhanced H-BitLinear layer implementing all BitNet v3 innovations.
    
    This is a drop-in replacement for nn.Linear that incorporates:
    - Progressive quantization during training
    - Adaptive Hadamard transform with learnable parameters
    - Enhanced straight-through estimation with momentum
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[BitLinearConfig] = None,
        layer_name: str = "bitlinear",
        **kwargs
    ):
        super().__init__()
        
        # Initialize configuration
        if config is None:
            config = BitLinearConfig(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                **kwargs
            )
        
        self.config = config
        self.layer_name = layer_name
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize sub-modules
        self._initialize_components()
        
        # Initialize parameters
        self.reset_parameters()
        
        # Statistics tracking
        if config.enable_statistics:
            self._initialize_statistics()
    
    def _initialize_components(self):
        """Initialize all component modules."""
        config = self.config
        
        # 1. Multi-stage Progressive Quantizer
        if config.enable_mpq:
            mpq_config = config.mpq_config or MPQConfig()
            self.mpq_quantizer = MultiStageProgressiveQuantizer(mpq_config)
        else:
            self.mpq_quantizer = None
        
        # 2. Adaptive Hadamard Transform
        if config.enable_adaptive_hadamard:
            # Determine input size for Hadamard transform
            hadamard_size = config.in_features
            if config.pad_to_power_of_2 and not is_power_of_2(hadamard_size):
                hadamard_size = 2 ** (hadamard_size - 1).bit_length()
            
            self.adaptive_hadamard = AdaptiveHadamardTransform(
                size=hadamard_size,
                normalize=config.hadamard_normalize,
                use_fast_transform=config.use_fast_hadamard,
                learnable_params=True,
                init_scale=config.hadamard_init_scale,
                init_shift=config.hadamard_init_shift,
                scale_lr_multiplier=config.hadamard_scale_lr_multiplier,
                shift_lr_multiplier=config.hadamard_shift_lr_multiplier,
            )
            self.hadamard_size = hadamard_size
        else:
            self.adaptive_hadamard = None
            self.hadamard_size = config.in_features
        
        # 3. Enhanced STE with Momentum
        if config.enable_enhanced_ste:
            self.enhanced_ste = EnhancedSTEMomentum(
                momentum=config.ste_momentum,
                power=config.ste_power,
                clip_value=config.ste_clip_value,
                enable_enhanced_ste=True,
            )
        else:
            self.enhanced_ste = None
    
    def _initialize_statistics(self):
        """Initialize statistics tracking buffers."""
        # Quantization statistics
        self.register_buffer('weight_quant_error', torch.tensor(0.0))
        self.register_buffer('activation_quant_error', torch.tensor(0.0))
        
        # Hadamard statistics
        self.register_buffer('hadamard_input_stats', torch.zeros(4))  # min, max, mean, std
        self.register_buffer('hadamard_output_stats', torch.zeros(4))
        
        # Step counter
        self.register_buffer('forward_steps', torch.tensor(0, dtype=torch.long))
    
    def reset_parameters(self):
        """Initialize layer parameters."""
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _update_statistics(
        self, 
        x_input: torch.Tensor,
        x_hadamard: torch.Tensor,
        weight_orig: torch.Tensor,
        weight_quant: torch.Tensor,
        x_quant: torch.Tensor
    ):
        """Update training statistics."""
        if not self.config.enable_statistics or not self.training:
            return
        
        with torch.no_grad():
            self.forward_steps += 1
            
            # Update quantization errors
            weight_error = torch.norm(weight_orig - weight_quant, p=2) / weight_orig.numel()
            self.weight_quant_error = 0.9 * self.weight_quant_error + 0.1 * weight_error
            
            if x_quant is not None:
                activation_error = torch.norm(x_hadamard - x_quant, p=2) / x_hadamard.numel()
                self.activation_quant_error = 0.9 * self.activation_quant_error + 0.1 * activation_error
            
            # Update Hadamard statistics
            self.hadamard_input_stats[0] = x_input.min()
            self.hadamard_input_stats[1] = x_input.max()
            self.hadamard_input_stats[2] = x_input.mean()
            self.hadamard_input_stats[3] = x_input.std()
            
            self.hadamard_output_stats[0] = x_hadamard.min()
            self.hadamard_output_stats[1] = x_hadamard.max()
            self.hadamard_output_stats[2] = x_hadamard.mean()
            self.hadamard_output_stats[3] = x_hadamard.std()
    
    def _apply_hadamard_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive Hadamard transform to input."""
        if not self.config.enable_adaptive_hadamard:
            return x
        
        # Pad input if necessary
        *batch_dims, input_size = x.shape
        if input_size != self.hadamard_size:
            if input_size < self.hadamard_size:
                # Pad to hadamard_size
                x = F.pad(x, (0, self.hadamard_size - input_size))
            else:
                # Truncate to hadamard_size
                x = x[..., :self.hadamard_size]
        
        # Apply adaptive Hadamard transform
        x_transformed = self.adaptive_hadamard(x)
        
        # Truncate back to original size if we padded
        if input_size < self.hadamard_size:
            x_transformed = x_transformed[..., :input_size]
        
        return x_transformed
    
    def _quantize_weights(self, epoch: Optional[int] = None) -> torch.Tensor:
        """Quantize weights using MPQ if enabled."""
        if self.config.enable_mpq and self.mpq_quantizer is not None:
            if epoch is not None:
                self.mpq_quantizer.set_epoch(epoch)
            return self.mpq_quantizer(self.weight, is_weight=True)
        else:
            # Direct quantization
            return quantize_weights_158(self.weight, self.config.weight_quantization_method)
    
    def _quantize_activations(self, x: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        """Quantize activations using MPQ if enabled."""
        if self.config.enable_mpq and self.mpq_quantizer is not None:
            if epoch is not None:
                self.mpq_quantizer.set_epoch(epoch)
            return self.mpq_quantizer(x, is_weight=False)
        else:
            # Direct quantization
            return quantize_activations(
                x, 
                self.config.activation_bits, 
                self.config.activation_quantization_method
            )
    
    def _apply_enhanced_ste(self, tensor: torch.Tensor, quantize_fn, is_weight: bool = True) -> torch.Tensor:
        """Apply enhanced STE with momentum."""
        if self.config.enable_enhanced_ste and self.enhanced_ste is not None:
            layer_suffix = "weight" if is_weight else "activation"
            layer_id = f"{self.layer_name}_{layer_suffix}"
            return self.enhanced_ste(tensor, quantize_fn, layer_id)
        else:
            # Standard straight-through estimator
            return quantize_fn(tensor)
    
    def forward(self, x: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with all BitNet v3 innovations.
        
        Args:
            x: Input tensor of shape (..., in_features)
            epoch: Current training epoch (for MPQ scheduling)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, x.size(-1))
        
        # 1. Apply Adaptive Hadamard Transform
        x_hadamard = self._apply_hadamard_transform(x_flat)
        
        # 2. Quantize activations with Enhanced STE
        if self.training:
            x_quantized = self._apply_enhanced_ste(
                x_hadamard,
                lambda tensor: self._quantize_activations(tensor, epoch),
                is_weight=False
            )
        else:
            x_quantized = self._quantize_activations(x_hadamard, epoch)
        
        # 3. Quantize weights with Enhanced STE  
        if self.training:
            weight_quantized = self._apply_enhanced_ste(
                self.weight,
                lambda tensor: self._quantize_weights(epoch),
                is_weight=True
            )
        else:
            weight_quantized = self._quantize_weights(epoch)
        
        # 4. Compute linear transformation
        output = F.linear(x_quantized, weight_quantized, self.bias)
        
        # 5. Update statistics
        if self.config.enable_statistics:
            self._update_statistics(
                x_flat, x_hadamard, self.weight, weight_quantized, x_quantized
            )
        
        # Reshape output
        return output.view(*batch_shape, self.out_features)
    
    def set_epoch(self, epoch: int):
        """Set current epoch for MPQ scheduling."""
        if self.mpq_quantizer is not None:
            self.mpq_quantizer.set_epoch(epoch)
    
    def get_quantization_status(self) -> Dict[str, Any]:
        """Get current quantization status and statistics."""
        status = {
            'layer_name': self.layer_name,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'config': {
                'weight_bits': self.config.weight_bits,
                'activation_bits': self.config.activation_bits,
                'enable_mpq': self.config.enable_mpq,
                'enable_adaptive_hadamard': self.config.enable_adaptive_hadamard,
                'enable_enhanced_ste': self.config.enable_enhanced_ste,
            }
        }
        
        # MPQ status
        if self.mpq_quantizer is not None:
            status['mpq_status'] = self.mpq_quantizer.get_current_status()
        
        # Adaptive Hadamard status
        if self.adaptive_hadamard is not None:
            status['hadamard_analysis'] = self.adaptive_hadamard.analyze_adaptation()
        
        # ESTE-M status
        if self.enhanced_ste is not None:
            status['este_m_metrics'] = self.enhanced_ste.get_performance_metrics()
        
        # Statistics
        if self.config.enable_statistics:
            status['statistics'] = {
                'forward_steps': self.forward_steps.item(),
                'weight_quant_error': self.weight_quant_error.item(),
                'activation_quant_error': self.activation_quant_error.item(),
                'hadamard_input_stats': {
                    'min': self.hadamard_input_stats[0].item(),
                    'max': self.hadamard_input_stats[1].item(),
                    'mean': self.hadamard_input_stats[2].item(),
                    'std': self.hadamard_input_stats[3].item(),
                },
                'hadamard_output_stats': {
                    'min': self.hadamard_output_stats[0].item(),
                    'max': self.hadamard_output_stats[1].item(),
                    'mean': self.hadamard_output_stats[2].item(),
                    'std': self.hadamard_output_stats[3].item(),
                },
            }
        
        return status
    
    def get_parameters_for_optimizer(self, base_lr: float) -> Dict[str, Dict[str, Any]]:
        """Get parameters with their specific learning rates for optimizer setup."""
        param_groups = {
            'main_params': {
                'params': [self.weight],
                'lr': base_lr,
            }
        }
        
        if self.bias is not None:
            param_groups['main_params']['params'].append(self.bias)
        
        # Add Adaptive Hadamard parameters with custom learning rates
        if self.adaptive_hadamard is not None:
            hadamard_params = self.adaptive_hadamard.get_parameters_for_optimizer(base_lr)
            param_groups.update(hadamard_params)
        
        return param_groups
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        features_str = f'in_features={self.in_features}, out_features={self.out_features}'
        bias_str = f', bias={self.bias is not None}'
        
        innovations = []
        if self.config.enable_mpq:
            innovations.append('MPQ')
        if self.config.enable_adaptive_hadamard:
            innovations.append('AHT-LP')
        if self.config.enable_enhanced_ste:
            innovations.append('ESTE-M')
        
        innovations_str = f', innovations=[{", ".join(innovations)}]'
        
        return features_str + bias_str + innovations_str


def replace_linear_with_bitlinear(
    module: nn.Module,
    config: Optional[BitLinearConfig] = None,
    target_layers: Optional[list] = None,
    exclude_layers: Optional[list] = None,
    **kwargs
) -> Dict[str, EnhancedHBitLinear]:
    """
    Replace all Linear layers in a module with EnhancedHBitLinear layers.
    
    Args:
        module: PyTorch module to modify
        config: Default configuration for BitLinear layers
        target_layers: List of layer names to replace (if None, replace all)
        exclude_layers: List of layer names to exclude from replacement
        **kwargs: Additional arguments for BitLinear configuration
        
    Returns:
        Dictionary mapping layer names to new BitLinear layers
    """
    exclude_layers = exclude_layers or []
    replaced_layers = {}
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Check if this layer should be replaced
            if target_layers is not None and name not in target_layers:
                continue
            if name in exclude_layers:
                continue
            
            # Create BitLinear configuration
            layer_config = config or BitLinearConfig(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                **kwargs
            )
            layer_config.in_features = child.in_features
            layer_config.out_features = child.out_features
            layer_config.bias = child.bias is not None
            
            # Create new BitLinear layer
            bitlinear_layer = EnhancedHBitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                config=layer_config,
                layer_name=name,
            )
            
            # Copy weights and bias
            with torch.no_grad():
                bitlinear_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    bitlinear_layer.bias.copy_(child.bias)
            
            # Replace the layer
            setattr(module, name, bitlinear_layer)
            replaced_layers[name] = bitlinear_layer
        
        # Recursively process child modules
        child_replacements = replace_linear_with_bitlinear(
            child, config, target_layers, exclude_layers, **kwargs
        )
        
        # Add child replacements with full path names
        for child_name, child_layer in child_replacements.items():
            full_name = f"{name}.{child_name}"
            replaced_layers[full_name] = child_layer
    
    return replaced_layers


def create_bitlinear_layer(
    in_features: int,
    out_features: int,
    bias: bool = False,
    enable_all_innovations: bool = True,
    **kwargs
) -> EnhancedHBitLinear:
    """
    Convenience function to create a BitLinear layer with sensible defaults.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias
        enable_all_innovations: Whether to enable all BitNet v3 innovations
        **kwargs: Additional configuration options
        
    Returns:
        Configured EnhancedHBitLinear layer
    """
    config = BitLinearConfig(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        enable_mpq=enable_all_innovations,
        enable_adaptive_hadamard=enable_all_innovations,
        enable_enhanced_ste=enable_all_innovations,
        **kwargs
    )
    
    return EnhancedHBitLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        config=config,
    )