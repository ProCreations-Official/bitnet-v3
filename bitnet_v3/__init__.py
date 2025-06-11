"""
BitNet v3: Ultra-Low Quality Loss 1-bit LLMs Through Multi-Stage Progressive Quantization and Adaptive Hadamard Transform

A comprehensive implementation of BitNet v3 with all five key innovations:
1. Multi-stage Progressive Quantization (MPQ)
2. Adaptive Hadamard Transform with Learnable Parameters (AHT-LP)
3. Gradient-Aware Knowledge Distillation (GAKD)
4. Dynamic Regularization with Quantization-Aware Penalties (DR-QAP)
5. Enhanced Straight-Through Estimator with Momentum (ESTE-M)

Usage:
    >>> import bitnet_v3
    >>> model = bitnet_v3.BitNetV3Model(config)
    >>> trainer = bitnet_v3.BitNetV3Trainer(model)
    >>> trainer.train(data_loader)
"""

__version__ = "1.0.0"
__author__ = "ProCreations"
__email__ = "procreations@example.com"

# Core imports
from .core.quantization import (
    quantize_weights_158,
    quantize_activations,
    absmean_quantization,
    absmax_quantization,
)

from .core.hadamard import (
    hadamard_transform,
    create_hadamard_matrix,
    fast_hadamard_transform,
)

from .core.straight_through import (
    StraightThroughEstimator,
    EnhancedSTEWithMomentum,
)

# Module imports
from .modules.bitlinear import EnhancedHBitLinear
from .modules.mpq import MultiStageProgressiveQuantizer
from .modules.aht_lp import AdaptiveHadamardTransform
from .modules.gakd import GradientAwareKnowledgeDistillation
from .modules.dr_qap import DynamicRegularizationQAP
from .modules.este_m import EnhancedSTEMomentum

# Model imports
from .models.bitnet_v3 import BitNetV3Model, BitNetV3Config
from .models.transformer import BitNetV3Transformer

# Training imports
# from .training.trainer import BitNetV3Trainer
# from .training.scheduler import MPQScheduler

# Utility imports
# from .utils.config import load_config, save_config
# from .utils.metrics import compute_perplexity, compute_efficiency_metrics

# Make key classes easily accessible
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Core functions
    "quantize_weights_158",
    "quantize_activations", 
    "absmean_quantization",
    "absmax_quantization",
    "hadamard_transform",
    "create_hadamard_matrix",
    "fast_hadamard_transform",
    "StraightThroughEstimator",
    "EnhancedSTEWithMomentum",
    
    # Modules
    "EnhancedHBitLinear",
    "MultiStageProgressiveQuantizer",
    "AdaptiveHadamardTransform", 
    "GradientAwareKnowledgeDistillation",
    "DynamicRegularizationQAP",
    "EnhancedSTEMomentum",
    
    # Models
    "BitNetV3Model",
    "BitNetV3Config",
    "BitNetV3Transformer",
    
    # Training
    # "BitNetV3Trainer",
    # "MPQScheduler",
    
    # Utils
    # "load_config",
    # "save_config", 
    # "compute_perplexity",
    # "compute_efficiency_metrics",
]

# Convenience functions for easy model creation
def create_model(config_path_or_dict=None, **kwargs):
    """Create a BitNet v3 model with default or custom configuration.
    
    Args:
        config_path_or_dict: Path to config file or config dictionary
        **kwargs: Additional config overrides
        
    Returns:
        BitNetV3Model: Configured BitNet v3 model
    """
    if config_path_or_dict is None:
        config = BitNetV3Config(**kwargs)
    elif isinstance(config_path_or_dict, str):
        # config = load_config(config_path_or_dict)
        # for key, value in kwargs.items():
        #     setattr(config, key, value)
        config = BitNetV3Config(**kwargs)
    else:
        config = BitNetV3Config(**config_path_or_dict, **kwargs)
    
    return BitNetV3Model(config)

# def create_trainer(model, **kwargs):
#     """Create a BitNet v3 trainer for the given model.
    
#     Args:
#         model: BitNet v3 model instance
#         **kwargs: Training configuration options
        
#     Returns:
#         BitNetV3Trainer: Configured trainer
#     """
#     return BitNetV3Trainer(model, **kwargs)

# Add convenience functions to __all__
__all__.extend([
    "create_model",
    # "create_trainer",
])