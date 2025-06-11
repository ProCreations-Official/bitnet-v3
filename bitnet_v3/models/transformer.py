"""
BitNet v3 Transformer implementation with all innovations integrated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, NamedTuple
import math

from ..modules.bitlinear import EnhancedHBitLinear, BitLinearConfig


class TransformerOutput(NamedTuple):
    """Output of transformer forward pass."""
    last_hidden_state: torch.Tensor
    past_key_values: Optional[List[Tuple]] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None


class BitNetV3Attention(nn.Module):
    """Multi-head attention with BitNet v3 innovations."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}")
        
        # Create BitLinear config for attention layers
        bitlinear_config = BitLinearConfig(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            enable_mpq=config.enable_mpq,
            enable_adaptive_hadamard=config.enable_adaptive_hadamard,
            enable_enhanced_ste=config.enable_enhanced_ste,
        )
        
        if config.enable_bitnet_innovations:
            # Use Enhanced H-BitLinear for all projections
            self.q_proj = EnhancedHBitLinear(
                self.hidden_size, self.hidden_size, 
                config=bitlinear_config, layer_name="q_proj"
            )
            self.k_proj = EnhancedHBitLinear(
                self.hidden_size, self.hidden_size,
                config=bitlinear_config, layer_name="k_proj" 
            )
            self.v_proj = EnhancedHBitLinear(
                self.hidden_size, self.hidden_size,
                config=bitlinear_config, layer_name="v_proj"
            )
            self.o_proj = EnhancedHBitLinear(
                self.hidden_size, self.hidden_size,
                config=bitlinear_config, layer_name="o_proj"
            )
        else:
            # Standard linear layers
            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass of attention layer."""
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        if isinstance(self.q_proj, EnhancedHBitLinear):
            query_states = self.q_proj(hidden_states, epoch)
            key_states = self.k_proj(hidden_states, epoch)
            value_states = self.v_proj(hidden_states, epoch)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key values for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # Update past key values
        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None
        
        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand mask dimensions for heads
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply causal mask for autoregressive generation
        seq_len_k = key_states.size(2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len_k, device=attention_scores.device), diagonal=1)
        attention_scores = attention_scores.masked_fill(causal_mask.bool(), float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, value_states)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_size)
        
        if isinstance(self.o_proj, EnhancedHBitLinear):
            attention_output = self.o_proj(attention_output, epoch)
        else:
            attention_output = self.o_proj(attention_output)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class BitNetV3MLP(nn.Module):
    """MLP block with BitNet v3 innovations."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Create BitLinear config for MLP layers
        gate_config = BitLinearConfig(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
            enable_mpq=config.enable_mpq,
            enable_adaptive_hadamard=config.enable_adaptive_hadamard,
            enable_enhanced_ste=config.enable_enhanced_ste,
        )
        
        up_config = BitLinearConfig(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
            enable_mpq=config.enable_mpq,
            enable_adaptive_hadamard=config.enable_adaptive_hadamard,
            enable_enhanced_ste=config.enable_enhanced_ste,
        )
        
        down_config = BitLinearConfig(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            enable_mpq=config.enable_mpq,
            enable_adaptive_hadamard=config.enable_adaptive_hadamard,
            enable_enhanced_ste=config.enable_enhanced_ste,
        )
        
        if config.enable_bitnet_innovations:
            # Use Enhanced H-BitLinear for all MLP layers
            self.gate_proj = EnhancedHBitLinear(
                self.hidden_size, self.intermediate_size,
                config=gate_config, layer_name="gate_proj"
            )
            self.up_proj = EnhancedHBitLinear(
                self.hidden_size, self.intermediate_size,
                config=up_config, layer_name="up_proj"
            )
            self.down_proj = EnhancedHBitLinear(
                self.intermediate_size, self.hidden_size,
                config=down_config, layer_name="down_proj"
            )
        else:
            # Standard linear layers
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.activation_fn = F.silu  # SiLU activation
    
    def forward(self, hidden_states: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        """Forward pass of MLP block."""
        if isinstance(self.gate_proj, EnhancedHBitLinear):
            gate = self.gate_proj(hidden_states, epoch)
            up = self.up_proj(hidden_states, epoch)
            down = self.down_proj(self.activation_fn(gate) * up, epoch)
        else:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
            down = self.down_proj(self.activation_fn(gate) * up)
        
        return down


class BitNetV3TransformerBlock(nn.Module):
    """Single transformer block with BitNet v3 innovations."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Self-attention
        self.self_attn = BitNetV3Attention(config)
        
        # MLP
        self.mlp = BitNetV3MLP(config)
        
        # Layer norms
        if config.rms_norm:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass of transformer block."""
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            epoch=epoch,
        )
        
        hidden_states = self_attn_outputs[0]
        hidden_states = self.hidden_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, epoch)
        hidden_states = self.hidden_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_outputs[1],)
        if use_cache:
            outputs += (self_attn_outputs[-1],)
        
        return outputs
    
    def get_bitnet_status(self) -> Dict[str, Any]:
        """Get BitNet v3 status for this layer."""
        status = {
            'layer_idx': self.layer_idx,
            'attention': {},
            'mlp': {},
        }
        
        # Attention layer status
        if hasattr(self.self_attn.q_proj, 'get_quantization_status'):
            status['attention']['q_proj'] = self.self_attn.q_proj.get_quantization_status()
            status['attention']['k_proj'] = self.self_attn.k_proj.get_quantization_status()
            status['attention']['v_proj'] = self.self_attn.v_proj.get_quantization_status()
            status['attention']['o_proj'] = self.self_attn.o_proj.get_quantization_status()
        
        # MLP layer status
        if hasattr(self.mlp.gate_proj, 'get_quantization_status'):
            status['mlp']['gate_proj'] = self.mlp.gate_proj.get_quantization_status()
            status['mlp']['up_proj'] = self.mlp.up_proj.get_quantization_status()
            status['mlp']['down_proj'] = self.mlp.down_proj.get_quantization_status()
        
        return status


class BitNetV3Transformer(nn.Module):
    """Full transformer with BitNet v3 innovations."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BitNetV3TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # Current epoch for MPQ
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Set current epoch for all layers."""
        self.current_epoch = epoch
        
        # Propagate to all layers that have BitLinear modules
        for layer in self.layers:
            # Set epoch for attention layers
            if hasattr(layer.self_attn.q_proj, 'set_epoch'):
                layer.self_attn.q_proj.set_epoch(epoch)
                layer.self_attn.k_proj.set_epoch(epoch)
                layer.self_attn.v_proj.set_epoch(epoch)
                layer.self_attn.o_proj.set_epoch(epoch)
            
            # Set epoch for MLP layers
            if hasattr(layer.mlp.gate_proj, 'set_epoch'):
                layer.mlp.gate_proj.set_epoch(epoch)
                layer.mlp.up_proj.set_epoch(epoch)
                layer.mlp.down_proj.set_epoch(epoch)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        epoch: Optional[int] = None,
    ) -> TransformerOutput:
        """Forward pass through all transformer layers."""
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        
        if epoch is not None:
            self.set_epoch(epoch)
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Process through all layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Get past key value for this layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # Forward pass through layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                epoch=self.current_epoch,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        
        # Add final hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return TransformerOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
    def get_bitnet_status(self) -> Dict[str, Any]:
        """Get comprehensive BitNet v3 status for all layers."""
        status = {
            'num_layers': len(self.layers),
            'current_epoch': self.current_epoch,
            'layers': {},
        }
        
        for idx, layer in enumerate(self.layers):
            status['layers'][f'layer_{idx}'] = layer.get_bitnet_status()
        
        return status


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