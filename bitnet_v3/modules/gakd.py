"""
Gradient-Aware Knowledge Distillation (GAKD) for BitNet v3.

Implements the novel distillation loss that considers both output distributions and gradient flow:
L_GAKD = α * L_KL(p_s || p_t) + β * L_grad + γ * L_feature

where:
- L_KL is the standard KL divergence between student and teacher outputs
- L_grad = ||∇_W L_s - ∇_W L_t||_2 measures gradient alignment
- L_feature = Σ_l ||F_l^s - F_l^t||_2 aligns intermediate features

This preserves critical gradient information during quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import warnings
from collections import defaultdict


class GradientAlignmentLoss(nn.Module):
    """
    Computes gradient alignment loss between student and teacher models.
    
    L_grad = ||∇_W L_s - ∇_W L_t||_2
    """
    
    def __init__(
        self,
        norm_type: int = 2,
        normalize_by_param_count: bool = True,
        gradient_clip_value: Optional[float] = None,
        weight_by_layer_size: bool = True,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.normalize_by_param_count = normalize_by_param_count
        self.gradient_clip_value = gradient_clip_value
        self.weight_by_layer_size = weight_by_layer_size
    
    def forward(
        self,
        student_gradients: Dict[str, torch.Tensor],
        teacher_gradients: Dict[str, torch.Tensor],
        layer_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute gradient alignment loss.
        
        Args:
            student_gradients: Dictionary of {layer_name: gradient_tensor}
            teacher_gradients: Dictionary of {layer_name: gradient_tensor}  
            layer_weights: Optional weights for different layers
            
        Returns:
            Gradient alignment loss
        """
        total_loss = 0.0
        total_params = 0
        matched_layers = 0
        
        for layer_name in student_gradients:
            if layer_name not in teacher_gradients:
                warnings.warn(f"Layer {layer_name} not found in teacher gradients")
                continue
                
            student_grad = student_gradients[layer_name]
            teacher_grad = teacher_gradients[layer_name]
            
            # Ensure same shape
            if student_grad.shape != teacher_grad.shape:
                warnings.warn(f"Shape mismatch for layer {layer_name}: "
                            f"{student_grad.shape} vs {teacher_grad.shape}")
                continue
            
            # Apply gradient clipping if specified
            if self.gradient_clip_value is not None:
                student_grad = student_grad.clamp(-self.gradient_clip_value, self.gradient_clip_value)
                teacher_grad = teacher_grad.clamp(-self.gradient_clip_value, self.gradient_clip_value)
            
            # Compute gradient difference
            grad_diff = student_grad - teacher_grad
            
            # Compute norm
            if self.norm_type == 2:
                layer_loss = torch.norm(grad_diff, p=2)
            elif self.norm_type == 1:
                layer_loss = torch.norm(grad_diff, p=1)
            else:
                layer_loss = torch.norm(grad_diff, p=self.norm_type)
            
            # Apply layer weighting
            if layer_weights is not None and layer_name in layer_weights:
                layer_loss = layer_loss * layer_weights[layer_name]
            elif self.weight_by_layer_size:
                # Weight by parameter count in layer
                param_count = student_grad.numel()
                layer_loss = layer_loss * math.sqrt(param_count)
            
            total_loss += layer_loss
            total_params += student_grad.numel()
            matched_layers += 1
        
        if matched_layers == 0:
            warnings.warn("No matching layers found for gradient alignment")
            return torch.tensor(0.0, device=next(iter(student_gradients.values())).device)
        
        # Normalize by parameter count if requested
        if self.normalize_by_param_count and total_params > 0:
            total_loss = total_loss / math.sqrt(total_params)
        
        return total_loss


class FeatureAlignmentLoss(nn.Module):
    """
    Computes feature alignment loss between intermediate representations.
    
    L_feature = Σ_l ||F_l^s - F_l^t||_2
    """
    
    def __init__(
        self,
        distance_type: str = "mse",  # "mse", "cosine", "kl"
        normalize_features: bool = True,
        temperature: float = 4.0,
        align_method: str = "direct",  # "direct", "projection", "attention"
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.distance_type = distance_type
        self.normalize_features = normalize_features
        self.temperature = temperature
        self.align_method = align_method
        
        # Projection layers for dimension alignment
        self.projections = nn.ModuleDict()
        self.projection_dim = projection_dim
    
    def _create_projection(self, student_dim: int, teacher_dim: int, layer_name: str):
        """Create projection layer to align feature dimensions."""
        if self.projection_dim is not None:
            # Project both to common dimension
            self.projections[f"{layer_name}_student"] = nn.Linear(student_dim, self.projection_dim)
            self.projections[f"{layer_name}_teacher"] = nn.Linear(teacher_dim, self.projection_dim)
        else:
            # Project student to teacher dimension
            if student_dim != teacher_dim:
                self.projections[f"{layer_name}_student"] = nn.Linear(student_dim, teacher_dim)
    
    def _align_features(
        self, 
        student_features: torch.Tensor, 
        teacher_features: torch.Tensor,
        layer_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align feature dimensions and normalize if needed."""
        # Handle dimension mismatch
        if student_features.shape[-1] != teacher_features.shape[-1]:
            # Create projection if it doesn't exist
            if f"{layer_name}_student" not in self.projections:
                self._create_projection(
                    student_features.shape[-1], 
                    teacher_features.shape[-1], 
                    layer_name
                )
            
            # Apply projections
            if f"{layer_name}_student" in self.projections:
                student_features = self.projections[f"{layer_name}_student"](student_features)
            if f"{layer_name}_teacher" in self.projections:
                teacher_features = self.projections[f"{layer_name}_teacher"](teacher_features)
        
        # Normalize features if requested
        if self.normalize_features:
            student_features = F.normalize(student_features, p=2, dim=-1)
            teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        
        return student_features, teacher_features
    
    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        layer_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute feature alignment loss.
        
        Args:
            student_features: Dictionary of {layer_name: feature_tensor}
            teacher_features: Dictionary of {layer_name: feature_tensor}
            layer_weights: Optional weights for different layers
            
        Returns:
            Feature alignment loss
        """
        total_loss = 0.0
        matched_layers = 0
        
        for layer_name in student_features:
            if layer_name not in teacher_features:
                warnings.warn(f"Layer {layer_name} not found in teacher features")
                continue
            
            student_feat = student_features[layer_name]
            teacher_feat = teacher_features[layer_name]
            
            # Align features
            student_aligned, teacher_aligned = self._align_features(
                student_feat, teacher_feat, layer_name
            )
            
            # Compute distance
            if self.distance_type == "mse":
                layer_loss = F.mse_loss(student_aligned, teacher_aligned)
            elif self.distance_type == "cosine":
                # Cosine similarity loss
                cosine_sim = F.cosine_similarity(
                    student_aligned.view(-1, student_aligned.size(-1)),
                    teacher_aligned.view(-1, teacher_aligned.size(-1)),
                    dim=1
                )
                layer_loss = 1 - cosine_sim.mean()
            elif self.distance_type == "kl":
                # KL divergence loss
                student_logits = student_aligned / self.temperature
                teacher_logits = teacher_aligned / self.temperature
                
                student_prob = F.log_softmax(student_logits, dim=-1)
                teacher_prob = F.softmax(teacher_logits, dim=-1)
                
                layer_loss = F.kl_div(student_prob, teacher_prob, reduction='batchmean')
            else:
                raise ValueError(f"Unknown distance type: {self.distance_type}")
            
            # Apply layer weighting
            if layer_weights is not None and layer_name in layer_weights:
                layer_loss = layer_loss * layer_weights[layer_name]
            
            total_loss += layer_loss
            matched_layers += 1
        
        if matched_layers == 0:
            warnings.warn("No matching layers found for feature alignment")
            return torch.tensor(0.0, device=next(iter(student_features.values())).device)
        
        return total_loss / matched_layers


class GAKDLoss(nn.Module):
    """
    Complete Gradient-Aware Knowledge Distillation loss.
    
    Combines output distillation, gradient alignment, and feature alignment.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,  # Weight for output distillation
        beta: float = 0.2,   # Weight for gradient alignment
        gamma: float = 0.1,  # Weight for feature alignment
        temperature: float = 4.0,
        gradient_alignment_config: Optional[Dict[str, Any]] = None,
        feature_alignment_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        self.temperature = temperature
        
        # Ensure weights sum to 1
        total_weight = alpha + beta + gamma
        if abs(total_weight - 1.0) > 1e-6:
            warnings.warn(f"GAKD weights sum to {total_weight}, not 1.0. Normalizing.")
            self.alpha = alpha / total_weight
            self.beta = beta / total_weight
            self.gamma = gamma / total_weight
        
        # Initialize component losses
        grad_config = gradient_alignment_config or {}
        self.gradient_loss = GradientAlignmentLoss(**grad_config)
        
        feat_config = feature_alignment_config or {}
        self.feature_loss = FeatureAlignmentLoss(**feat_config)
    
    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        student_gradients: Optional[Dict[str, torch.Tensor]] = None,
        teacher_gradients: Optional[Dict[str, torch.Tensor]] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        reduction: str = "batchmean",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GAKD loss.
        
        Args:
            student_outputs: Student model outputs (logits)
            teacher_outputs: Teacher model outputs (logits)
            student_gradients: Optional student gradients
            teacher_gradients: Optional teacher gradients
            student_features: Optional student intermediate features
            teacher_features: Optional teacher intermediate features
            reduction: Reduction method for KL loss
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # 1. Output distillation (KL divergence)
        if self.alpha > 0:
            student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=-1)
            
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
            kl_loss = kl_loss * (self.temperature ** 2)  # Scale by temperature squared
            
            losses['kl_loss'] = kl_loss
        else:
            losses['kl_loss'] = torch.tensor(0.0, device=student_outputs.device)
        
        # 2. Gradient alignment
        if self.beta > 0 and student_gradients is not None and teacher_gradients is not None:
            grad_loss = self.gradient_loss(student_gradients, teacher_gradients)
            losses['gradient_loss'] = grad_loss
        else:
            losses['gradient_loss'] = torch.tensor(0.0, device=student_outputs.device)
        
        # 3. Feature alignment
        if self.gamma > 0 and student_features is not None and teacher_features is not None:
            feat_loss = self.feature_loss(student_features, teacher_features)
            losses['feature_loss'] = feat_loss
        else:
            losses['feature_loss'] = torch.tensor(0.0, device=student_outputs.device)
        
        # 4. Total GAKD loss
        total_loss = (self.alpha * losses['kl_loss'] + 
                     self.beta * losses['gradient_loss'] + 
                     self.gamma * losses['feature_loss'])
        
        losses['total_loss'] = total_loss
        losses['weighted_kl'] = self.alpha * losses['kl_loss']
        losses['weighted_gradient'] = self.beta * losses['gradient_loss']
        losses['weighted_feature'] = self.gamma * losses['feature_loss']
        
        return losses
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss component weights."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
        }
    
    def set_loss_weights(self, alpha: float, beta: float, gamma: float):
        """Update loss component weights."""
        total = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total


class GradientAwareKnowledgeDistillation(nn.Module):
    """
    Main GAKD module with additional utilities for training integration.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        temperature: float = 4.0,
        adaptive_weights: bool = False,
        weight_adaptation_frequency: int = 100,
        gradient_accumulation_steps: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.gakd_loss = GAKDLoss(alpha, beta, gamma, temperature, **kwargs)
        self.adaptive_weights = adaptive_weights
        self.weight_adaptation_frequency = weight_adaptation_frequency
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Statistics tracking
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.loss_history = defaultdict(list)
        
        # Gradient and feature hooks
        self.gradient_hooks = {}
        self.feature_hooks = {}
        self.collected_gradients = {}
        self.collected_features = {}
    
    def register_gradient_hooks(self, model: nn.Module, layer_names: List[str]):
        """Register hooks to collect gradients from specified layers."""
        def create_grad_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.collected_gradients[name] = grad_output[0].detach().clone()
            return hook
        
        for name in layer_names:
            module = dict(model.named_modules())[name]
            handle = module.register_backward_hook(create_grad_hook(name))
            self.gradient_hooks[name] = handle
    
    def register_feature_hooks(self, model: nn.Module, layer_names: List[str]):
        """Register hooks to collect features from specified layers."""
        def create_feat_hook(name):
            def hook(module, input, output):
                self.collected_features[name] = output.detach().clone()
            return hook
        
        for name in layer_names:
            module = dict(model.named_modules())[name]
            handle = module.register_forward_hook(create_feat_hook(name))
            self.feature_hooks[name] = handle
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.gradient_hooks.values():
            handle.remove()
        for handle in self.feature_hooks.values():
            handle.remove()
        self.gradient_hooks.clear()
        self.feature_hooks.clear()
    
    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        student_model: Optional[nn.Module] = None,
        teacher_model: Optional[nn.Module] = None,
        use_collected_gradients: bool = True,
        use_collected_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GAKD loss with automatic gradient and feature collection.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            student_model: Optional student model for gradient collection
            teacher_model: Optional teacher model for gradient collection
            use_collected_gradients: Whether to use hook-collected gradients
            use_collected_features: Whether to use hook-collected features
            
        Returns:
            Dictionary of losses
        """
        # Use collected gradients and features if available
        student_gradients = self.collected_gradients if use_collected_gradients else None
        teacher_gradients = None  # Teacher gradients collected separately
        student_features = self.collected_features if use_collected_features else None
        teacher_features = None   # Teacher features collected separately
        
        # Compute GAKD loss
        losses = self.gakd_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            student_gradients=student_gradients,
            teacher_gradients=teacher_gradients,
            student_features=student_features,
            teacher_features=teacher_features,
        )
        
        # Update statistics
        self.step_count += 1
        for key, value in losses.items():
            self.loss_history[key].append(value.item())
        
        # Adaptive weight adjustment
        if (self.adaptive_weights and 
            self.step_count % self.weight_adaptation_frequency == 0):
            self._adapt_loss_weights()
        
        # Clear collected data
        self.collected_gradients.clear()
        self.collected_features.clear()
        
        return losses
    
    def _adapt_loss_weights(self):
        """Adapt loss weights based on training history."""
        if len(self.loss_history['kl_loss']) < 10:
            return  # Need enough history
        
        # Simple adaptation: balance losses to similar magnitudes
        recent_kl = sum(self.loss_history['kl_loss'][-10:]) / 10
        recent_grad = sum(self.loss_history['gradient_loss'][-10:]) / 10
        recent_feat = sum(self.loss_history['feature_loss'][-10:]) / 10
        
        # Avoid division by zero
        recent_grad = max(recent_grad, 1e-8)
        recent_feat = max(recent_feat, 1e-8)
        
        # Compute relative scales
        grad_scale = recent_kl / recent_grad
        feat_scale = recent_kl / recent_feat
        
        # Update weights (simple moving average)
        momentum = 0.9
        current_weights = self.gakd_loss.get_loss_weights()
        
        new_beta = momentum * current_weights['beta'] + (1 - momentum) * (grad_scale * current_weights['beta'])
        new_gamma = momentum * current_weights['gamma'] + (1 - momentum) * (feat_scale * current_weights['gamma'])
        new_alpha = 1.0 - new_beta - new_gamma
        
        # Ensure positive weights
        new_alpha = max(0.1, new_alpha)
        new_beta = max(0.05, new_beta) 
        new_gamma = max(0.05, new_gamma)
        
        self.gakd_loss.set_loss_weights(new_alpha, new_beta, new_gamma)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics and loss history."""
        return {
            'step_count': self.step_count.item(),
            'current_weights': self.gakd_loss.get_loss_weights(),
            'loss_history': dict(self.loss_history),
            'average_losses': {
                key: sum(values[-100:]) / len(values[-100:]) if values else 0.0
                for key, values in self.loss_history.items()
            },
        }
    
    def extra_repr(self) -> str:
        weights = self.gakd_loss.get_loss_weights()
        return (f"alpha={weights['alpha']:.3f}, beta={weights['beta']:.3f}, "
                f"gamma={weights['gamma']:.3f}, adaptive={self.adaptive_weights}")