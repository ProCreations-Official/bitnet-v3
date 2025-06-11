"""
Main BitNet v3 model implementations.

Complete model architectures that integrate all five BitNet v3 innovations.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import json
import warnings

from ..modules.bitlinear import EnhancedHBitLinear, BitLinearConfig
from ..modules.mpq import MPQConfig
from .transformer import BitNetV3Transformer


@dataclass
class BitNetV3Config:
    """Configuration for BitNet v3 models."""
    
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 32
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 2048
    
    # Attention configuration
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Normalization
    layer_norm_eps: float = 1e-5
    rms_norm: bool = True
    
    # BitNet v3 specific configurations
    enable_bitnet_innovations: bool = True
    
    # MPQ configuration
    enable_mpq: bool = True
    mpq_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"start_epoch": 1, "end_epoch": 20, "source_bits": 16, "target_bits": 8},
        {"start_epoch": 21, "end_epoch": 40, "source_bits": 8, "target_bits": 4},
        {"start_epoch": 41, "end_epoch": 55, "source_bits": 4, "target_bits": 2},
        {"start_epoch": 56, "end_epoch": 70, "source_bits": 2, "target_bits": 1.58},
    ])
    
    # AHT-LP configuration
    enable_adaptive_hadamard: bool = True
    hadamard_learnable_scale: bool = True
    hadamard_learnable_shift: bool = True
    
    # GAKD configuration
    enable_knowledge_distillation: bool = True
    gakd_alpha: float = 0.7
    gakd_beta: float = 0.2
    gakd_gamma: float = 0.1
    
    # DR-QAP configuration
    enable_dynamic_regularization: bool = True
    qap_initial_lambda: float = 0.1
    qap_decay_factor: float = 0.95
    
    # ESTE-M configuration
    enable_enhanced_ste: bool = True
    ste_momentum: float = 0.9
    ste_power: float = 0.5
    
    # Training configuration
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        
        # Validate head configuration
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'attention_dropout': self.attention_dropout,
            'hidden_dropout': self.hidden_dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'rms_norm': self.rms_norm,
            'enable_bitnet_innovations': self.enable_bitnet_innovations,
            'enable_mpq': self.enable_mpq,
            'mpq_stages': self.mpq_stages,
            'enable_adaptive_hadamard': self.enable_adaptive_hadamard,
            'hadamard_learnable_scale': self.hadamard_learnable_scale,
            'hadamard_learnable_shift': self.hadamard_learnable_shift,
            'enable_knowledge_distillation': self.enable_knowledge_distillation,
            'gakd_alpha': self.gakd_alpha,
            'gakd_beta': self.gakd_beta,
            'gakd_gamma': self.gakd_gamma,
            'enable_dynamic_regularization': self.enable_dynamic_regularization,
            'qap_initial_lambda': self.qap_initial_lambda,
            'qap_decay_factor': self.qap_decay_factor,
            'enable_enhanced_ste': self.enable_enhanced_ste,
            'ste_momentum': self.ste_momentum,
            'ste_power': self.ste_power,
            'initializer_range': self.initializer_range,
            'tie_word_embeddings': self.tie_word_embeddings,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BitNetV3Config':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: str):
        """Save configuration to directory."""
        import os
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'BitNetV3Config':
        """Load configuration from directory."""
        import os
        config_file = os.path.join(model_path, "config.json")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BitNetV3Model(nn.Module):
    """
    Main BitNet v3 model implementing all five key innovations.
    
    This is the base model that can be extended for specific tasks.
    """
    
    def __init__(self, config: BitNetV3Config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings (if needed)
        if config.max_position_embeddings > 0:
            self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_positions = None
        
        # Core transformer
        self.transformer = BitNetV3Transformer(config)
        
        # Final layer norm
        if config.rms_norm:
            self.final_layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Current epoch for MPQ scheduling
        self.current_epoch = 0
    
    def _init_weights(self, module):
        """Initialize weights according to BitNet v3 paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, EnhancedHBitLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def set_epoch(self, epoch: int):
        """Set current epoch for MPQ scheduling."""
        self.current_epoch = epoch
        # Propagate to all transformer layers
        self.transformer.set_epoch(epoch)
    
    def get_input_embeddings(self):
        """Get input token embeddings."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        """Set input token embeddings."""
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """
        Forward pass through BitNet v3 model.
        
        Args:
            input_ids: Token indices of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            position_ids: Position indices of shape (batch_size, sequence_length)
            past_key_values: Cached key-value pairs for generation
            use_cache: Whether to return key-value cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs (last hidden state, optionally with additional outputs)
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Position embeddings
        if self.embed_positions is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.embed_positions(position_ids)
            inputs_embeds = inputs_embeds + position_embeds
        
        # Transformer forward pass
        transformer_outputs = self.transformer(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            epoch=self.current_epoch,
        )
        
        # Final layer norm
        last_hidden_state = self.final_layer_norm(transformer_outputs.last_hidden_state)
        
        if return_dict:
            return {
                'last_hidden_state': last_hidden_state,
                'past_key_values': transformer_outputs.past_key_values,
                'hidden_states': transformer_outputs.hidden_states,
                'attentions': transformer_outputs.attentions,
            }
        else:
            return (
                last_hidden_state,
                transformer_outputs.past_key_values,
                transformer_outputs.hidden_states,
                transformer_outputs.attentions,
            )
    
    def get_bitnet_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all BitNet v3 innovations."""
        return self.transformer.get_bitnet_status()
    
    def save_pretrained(self, save_directory: str):
        """Save model and configuration."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Save model state
        model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_file)
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'BitNetV3Model':
        """Load model from directory."""
        import os
        
        # Load configuration
        config = BitNetV3Config.from_pretrained(model_path)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(state_dict)
        
        return model


class BitNetV3ForCausalLM(BitNetV3Model):
    """BitNet v3 model for causal language modeling."""
    
    def __init__(self, config: BitNetV3Config):
        super().__init__(config)
        
        # Language modeling head
        if config.enable_bitnet_innovations:
            # Use BitLinear for the LM head as well
            bitlinear_config = BitLinearConfig(
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                bias=False,
                enable_mpq=config.enable_mpq,
                enable_adaptive_hadamard=config.enable_adaptive_hadamard,
                enable_enhanced_ste=config.enable_enhanced_ste,
            )
            self.lm_head = EnhancedHBitLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                config=bitlinear_config,
                layer_name="lm_head"
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass with language modeling head."""
        return_dict = return_dict if return_dict is not None else True
        
        # Get transformer outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Compute logits
        if isinstance(self.lm_head, EnhancedHBitLinear):
            logits = self.lm_head(outputs['last_hidden_state'], epoch=self.current_epoch)
        else:
            logits = self.lm_head(outputs['last_hidden_state'])
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'past_key_values': outputs['past_key_values'],
                'hidden_states': outputs['hidden_states'],
                'attentions': outputs['attentions'],
            }
        else:
            return (loss, logits) + outputs[1:]
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        device = input_ids.device
        
        # Generation loop
        past_key_values = None
        
        for _ in range(max_length - current_length):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Get next token logits
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply sampling
            if do_sample:
                if top_k is not None:
                    # Top-k sampling
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Update past_key_values
            past_key_values = outputs['past_key_values']
            
            # Check for EOS token
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        return input_ids


class BitNetV3ForSequenceClassification(BitNetV3Model):
    """BitNet v3 model for sequence classification."""
    
    def __init__(self, config: BitNetV3Config, num_labels: int = 2):
        super().__init__(config)
        self.num_labels = num_labels
        
        # Classification head
        if config.enable_bitnet_innovations:
            bitlinear_config = BitLinearConfig(
                in_features=config.hidden_size,
                out_features=num_labels,
                bias=True,
                enable_mpq=config.enable_mpq,
                enable_adaptive_hadamard=config.enable_adaptive_hadamard,
                enable_enhanced_ste=config.enable_enhanced_ste,
            )
            self.classifier = EnhancedHBitLinear(
                config.hidden_size,
                num_labels,
                bias=True,
                config=bitlinear_config,
                layer_name="classifier"
            )
        else:
            self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass for sequence classification."""
        return_dict = return_dict if return_dict is not None else True
        
        # Get transformer outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Pool the sequence (use last token or mean pooling)
        last_hidden_state = outputs['last_hidden_state']
        
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            pooled_output = last_hidden_state.mean(dim=1)
        
        # Classification logits
        if isinstance(self.classifier, EnhancedHBitLinear):
            logits = self.classifier(pooled_output, epoch=self.current_epoch)
        else:
            logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': outputs['hidden_states'],
                'attentions': outputs['attentions'],
            }
        else:
            return (loss, logits) + outputs[1:]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)