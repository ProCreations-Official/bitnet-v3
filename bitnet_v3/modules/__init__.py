"""
BitNet v3 innovation modules.

This package contains implementations of the five key innovations in BitNet v3:
1. Multi-stage Progressive Quantization (MPQ)
2. Adaptive Hadamard Transform with Learnable Parameters (AHT-LP)  
3. Gradient-Aware Knowledge Distillation (GAKD)
4. Dynamic Regularization with Quantization-Aware Penalties (DR-QAP)
5. Enhanced Straight-Through Estimator with Momentum (ESTE-M)
"""

from .mpq import (
    MultiStageProgressiveQuantizer,
    MPQSchedule,
    MPQConfig,
    create_mpq_schedule,
)

from .aht_lp import (
    AdaptiveHadamardTransform,
    LearnableHadamardParameters,
)

from .gakd import (
    GradientAwareKnowledgeDistillation,
    GAKDLoss,
    FeatureAlignmentLoss,
    GradientAlignmentLoss,
)

from .dr_qap import (
    DynamicRegularizationQAP,
    QuantizationAwarePenalty,
    LayerSensitivityCalculator,
)

from .este_m import (
    EnhancedSTEMomentum,
    MomentumGradientEstimator,
)

from .bitlinear import (
    EnhancedHBitLinear,
    BitLinearConfig,
)

__all__ = [
    # MPQ
    "MultiStageProgressiveQuantizer",
    "MPQSchedule", 
    "MPQConfig",
    "create_mpq_schedule",
    
    # AHT-LP
    "AdaptiveHadamardTransform",
    "LearnableHadamardParameters",
    
    # GAKD
    "GradientAwareKnowledgeDistillation",
    "GAKDLoss",
    "FeatureAlignmentLoss",
    "GradientAlignmentLoss",
    
    # DR-QAP
    "DynamicRegularizationQAP",
    "QuantizationAwarePenalty",
    "LayerSensitivityCalculator",
    
    # ESTE-M
    "EnhancedSTEMomentum",
    "MomentumGradientEstimator",
    
    # BitLinear
    "EnhancedHBitLinear",
    "BitLinearConfig",
]