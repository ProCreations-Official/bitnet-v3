"""
Multi-stage Progressive Quantization (MPQ) implementation for BitNet v3.

Implements the progressive quantization scheme that gradually reduces bit-width during training:
Stage 1: FP16 → 8-bit (epochs 1-20)
Stage 2: 8-bit → 4-bit (epochs 21-40)  
Stage 3: 4-bit → 2-bit (epochs 41-55)
Stage 4: 2-bit → 1.58-bit (epochs 56-70)

Uses temperature-based transition function:
Q_t(x) = σ(β_t) · Q_{b_t}(x) + (1 - σ(β_t)) · Q_{b_{t-1}}(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import math
import warnings

from ..core.quantization import progressive_quantize, quantize_weights_158, quantize_activations


@dataclass
class MPQStage:
    """Configuration for a single MPQ stage."""
    start_epoch: int
    end_epoch: int
    source_bits: Union[int, float]
    target_bits: Union[int, float]
    temperature_schedule: str = "linear"  # "linear", "cosine", "exponential"
    initial_temperature: float = 0.0
    final_temperature: float = 10.0


@dataclass  
class MPQConfig:
    """Configuration for Multi-stage Progressive Quantization."""
    stages: List[MPQStage] = field(default_factory=lambda: [
        MPQStage(1, 20, 16, 8),      # FP16 to 8-bit
        MPQStage(21, 40, 8, 4),      # 8-bit to 4-bit  
        MPQStage(41, 55, 4, 2),      # 4-bit to 2-bit
        MPQStage(56, 70, 2, 1.58),   # 2-bit to 1.58-bit
    ])
    
    # Global settings
    warmup_epochs: int = 5
    use_soft_transitions: bool = True
    transition_smoothness: float = 2.0
    
    # Weight vs activation settings
    separate_weight_activation_schedule: bool = True
    weight_lag_epochs: int = 2  # Weights quantize 2 epochs after activations


class MPQSchedule:
    """
    Schedule manager for Multi-stage Progressive Quantization.
    """
    
    def __init__(self, config: MPQConfig):
        self.config = config
        self.current_epoch = 0
        self.current_stage_idx = 0
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate MPQ configuration."""
        stages = self.config.stages
        
        # Check epoch continuity
        for i in range(len(stages) - 1):
            if stages[i].end_epoch >= stages[i + 1].start_epoch:
                warnings.warn(f"Stage {i} end epoch {stages[i].end_epoch} "
                            f"overlaps with stage {i+1} start epoch {stages[i+1].start_epoch}")
        
        # Check bit-width progression
        for i in range(len(stages) - 1):
            if stages[i].target_bits <= stages[i + 1].target_bits:
                warnings.warn(f"Bit-width should decrease across stages: "
                            f"Stage {i}: {stages[i].target_bits} -> Stage {i+1}: {stages[i+1].target_bits}")
    
    def get_current_stage(self, epoch: int) -> Optional[MPQStage]:
        """Get the current MPQ stage for given epoch."""
        self.current_epoch = epoch
        
        for i, stage in enumerate(self.config.stages):
            if stage.start_epoch <= epoch <= stage.end_epoch:
                self.current_stage_idx = i
                return stage
        
        # After all stages, use final quantization
        if epoch > self.config.stages[-1].end_epoch:
            return None
            
        # Before first stage, use full precision
        return None
    
    def get_quantization_bits(self, epoch: int, is_weight: bool = True) -> Tuple[Union[int, float], Union[int, float], float]:
        """
        Get current quantization bit-widths and temperature.
        
        Args:
            epoch: Current training epoch
            is_weight: Whether this is for weights (vs activations)
            
        Returns:
            Tuple of (source_bits, target_bits, temperature)
        """
        stage = self.get_current_stage(epoch)
        
        if stage is None:
            # Return final quantization settings
            if epoch > self.config.stages[-1].end_epoch:
                final_bits = self.config.stages[-1].target_bits
                return final_bits, final_bits, 10.0
            else:
                # Before training starts, use full precision
                return 16, 16, 0.0
        
        # Apply weight lag if configured
        if (is_weight and 
            self.config.separate_weight_activation_schedule and 
            self.config.weight_lag_epochs > 0):
            
            adjusted_epoch = max(stage.start_epoch, epoch - self.config.weight_lag_epochs)
            if adjusted_epoch > stage.end_epoch:
                # Weight quantization for this stage is complete
                return stage.target_bits, stage.target_bits, 10.0
        else:
            adjusted_epoch = epoch
        
        # Calculate temperature based on progress through stage
        progress = (adjusted_epoch - stage.start_epoch) / max(1, stage.end_epoch - stage.start_epoch)
        progress = max(0.0, min(1.0, progress))
        
        temperature = self._calculate_temperature(stage, progress)
        
        return stage.source_bits, stage.target_bits, temperature
    
    def _calculate_temperature(self, stage: MPQStage, progress: float) -> float:
        """Calculate temperature based on schedule type and progress."""
        initial_temp = stage.initial_temperature
        final_temp = stage.final_temperature
        
        if stage.temperature_schedule == "linear":
            return initial_temp + progress * (final_temp - initial_temp)
        
        elif stage.temperature_schedule == "cosine":
            # Cosine annealing
            return initial_temp + 0.5 * (final_temp - initial_temp) * (1 + math.cos(math.pi * (1 - progress)))
        
        elif stage.temperature_schedule == "exponential":
            # Exponential growth
            if final_temp > initial_temp:
                ratio = final_temp / max(initial_temp, 1e-6)
                return initial_temp * (ratio ** progress)
            else:
                ratio = initial_temp / max(final_temp, 1e-6)
                return initial_temp / (ratio ** progress)
        
        else:
            raise ValueError(f"Unknown temperature schedule: {stage.temperature_schedule}")
    
    def is_transition_complete(self, epoch: int) -> bool:
        """Check if all MPQ transitions are complete."""
        return epoch > self.config.stages[-1].end_epoch
    
    def get_stage_info(self, epoch: int) -> Dict[str, Any]:
        """Get detailed information about current stage."""
        stage = self.get_current_stage(epoch)
        
        if stage is None:
            if epoch > self.config.stages[-1].end_epoch:
                return {
                    "stage": "complete",
                    "stage_idx": len(self.config.stages),
                    "bits": self.config.stages[-1].target_bits,
                    "progress": 1.0,
                }
            else:
                return {
                    "stage": "warmup",
                    "stage_idx": -1,
                    "bits": 16,
                    "progress": 0.0,
                }
        
        progress = (epoch - stage.start_epoch) / max(1, stage.end_epoch - stage.start_epoch)
        
        return {
            "stage": f"stage_{self.current_stage_idx}",
            "stage_idx": self.current_stage_idx,
            "source_bits": stage.source_bits,
            "target_bits": stage.target_bits,
            "progress": max(0.0, min(1.0, progress)),
            "epoch_range": (stage.start_epoch, stage.end_epoch),
        }


class MultiStageProgressiveQuantizer(nn.Module):
    """
    Multi-stage Progressive Quantizer implementing the MPQ scheme from BitNet v3.
    """
    
    def __init__(self, config: MPQConfig):
        super().__init__()
        self.config = config
        self.schedule = MPQSchedule(config)
        self.current_epoch = 0
        
        # Statistics tracking
        self.register_buffer('quantization_stats', torch.zeros(4))  # [min, max, mean, std]
        self.register_buffer('temperature_history', torch.zeros(100))  # Rolling history
        self.history_idx = 0
        
    def set_epoch(self, epoch: int):
        """Set current epoch for schedule calculation."""
        self.current_epoch = epoch
        
    def forward(
        self, 
        x: torch.Tensor, 
        is_weight: bool = True,
        force_bits: Optional[Union[int, float]] = None
    ) -> torch.Tensor:
        """
        Apply progressive quantization to input tensor.
        
        Args:
            x: Input tensor to quantize
            is_weight: Whether this is a weight tensor (vs activation)
            force_bits: Override automatic bit-width selection
            
        Returns:
            Progressively quantized tensor
        """
        if force_bits is not None:
            # Force specific quantization level
            if is_weight:
                return quantize_weights_158(x) if force_bits == 1.58 else quantize_activations(x, int(force_bits))
            else:
                return quantize_activations(x, int(force_bits))
        
        # Get current quantization parameters
        source_bits, target_bits, temperature = self.schedule.get_quantization_bits(
            self.current_epoch, is_weight
        )
        
        # Update statistics
        if self.training:
            self._update_stats(x, temperature)
        
        # Apply progressive quantization
        if source_bits == target_bits:
            # No transition needed
            if is_weight:
                return quantize_weights_158(x) if target_bits == 1.58 else quantize_activations(x, int(target_bits))
            else:
                return quantize_activations(x, int(target_bits))
        else:
            # Progressive transition
            return progressive_quantize(x, source_bits, target_bits, temperature, is_weight)
    
    def _update_stats(self, x: torch.Tensor, temperature: float):
        """Update quantization statistics."""
        with torch.no_grad():
            self.quantization_stats[0] = x.min()
            self.quantization_stats[1] = x.max()
            self.quantization_stats[2] = x.mean()
            self.quantization_stats[3] = x.std()
            
            # Update temperature history
            self.temperature_history[self.history_idx % 100] = temperature
            self.history_idx += 1
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current quantization status."""
        stage_info = self.schedule.get_stage_info(self.current_epoch)
        
        weight_bits = self.schedule.get_quantization_bits(self.current_epoch, is_weight=True)
        activation_bits = self.schedule.get_quantization_bits(self.current_epoch, is_weight=False)
        
        return {
            "epoch": self.current_epoch,
            "stage_info": stage_info,
            "weight_quantization": {
                "source_bits": weight_bits[0],
                "target_bits": weight_bits[1], 
                "temperature": weight_bits[2],
            },
            "activation_quantization": {
                "source_bits": activation_bits[0],
                "target_bits": activation_bits[1],
                "temperature": activation_bits[2],
            },
            "statistics": {
                "min": self.quantization_stats[0].item(),
                "max": self.quantization_stats[1].item(),
                "mean": self.quantization_stats[2].item(),
                "std": self.quantization_stats[3].item(),
            },
        }
    
    def extra_repr(self) -> str:
        return f'stages={len(self.config.stages)}, current_epoch={self.current_epoch}'


def create_mpq_schedule(
    total_epochs: int = 70,
    num_stages: int = 4,
    final_bits: float = 1.58,
    custom_stages: Optional[List[Dict[str, Any]]] = None
) -> MPQConfig:
    """
    Create a default MPQ schedule configuration.
    
    Args:
        total_epochs: Total training epochs
        num_stages: Number of quantization stages
        final_bits: Final bit-width (typically 1.58)
        custom_stages: Custom stage definitions
        
    Returns:
        MPQConfig object
    """
    if custom_stages is not None:
        stages = [MPQStage(**stage) for stage in custom_stages]
    else:
        # Create default schedule
        stage_epochs = total_epochs // num_stages
        bit_levels = [16, 8, 4, 2, final_bits]
        
        stages = []
        for i in range(num_stages):
            start_epoch = i * stage_epochs + 1
            end_epoch = (i + 1) * stage_epochs
            
            if i == num_stages - 1:
                end_epoch = total_epochs  # Ensure we cover all epochs
            
            stages.append(MPQStage(
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                source_bits=bit_levels[i],
                target_bits=bit_levels[i + 1],
                temperature_schedule="linear",
                initial_temperature=0.0,
                final_temperature=10.0,
            ))
    
    return MPQConfig(stages=stages)


class MPQTrainingCallback:
    """
    Training callback for MPQ integration with training loops.
    """
    
    def __init__(self, mpq_quantizer: MultiStageProgressiveQuantizer):
        self.mpq_quantizer = mpq_quantizer
        self.epoch_logs = []
    
    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch."""
        self.mpq_quantizer.set_epoch(epoch)
        
        # Log current status
        status = self.mpq_quantizer.get_current_status()
        self.epoch_logs.append(status)
        
        return status
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        status = self.mpq_quantizer.get_current_status()
        
        if logs is not None:
            # Add MPQ info to training logs
            logs.update({
                "mpq_stage": status["stage_info"]["stage"],
                "mpq_progress": status["stage_info"]["progress"],
                "weight_bits": status["weight_quantization"]["target_bits"],
                "activation_bits": status["activation_quantization"]["target_bits"],
                "mpq_temperature": status["weight_quantization"]["temperature"],
            })
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of MPQ training progress."""
        return {
            "total_epochs": len(self.epoch_logs),
            "stages_completed": len([log for log in self.epoch_logs 
                                   if log["stage_info"]["progress"] >= 1.0]),
            "final_bits": self.epoch_logs[-1]["weight_quantization"]["target_bits"] if self.epoch_logs else None,
            "epoch_logs": self.epoch_logs,
        }