"""
Basic usage examples for BitNet v3.

This demonstrates how to use the BitNet v3 package for creating and training models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import bitnet_v3


def create_simple_model():
    """Create a simple BitNet v3 model."""
    print("Creating BitNet v3 model...")
    
    # Create model configuration
    config = bitnet_v3.BitNetV3Config(
        vocab_size=1000,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        max_position_embeddings=128,
    )
    
    # Create model
    model = bitnet_v3.BitNetV3ForCausalLM(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def demonstrate_bitlinear_layer():
    """Demonstrate the Enhanced H-BitLinear layer."""
    print("\nDemonstrating Enhanced H-BitLinear layer...")
    
    # Create a BitLinear layer with all innovations enabled
    layer = bitnet_v3.create_bitlinear_layer(
        in_features=256,
        out_features=512,
        enable_all_innovations=True,
    )
    
    # Test forward pass
    x = torch.randn(2, 32, 256)  # batch_size=2, seq_len=32, features=256
    output = layer(x, epoch=1)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get quantization status
    status = layer.get_quantization_status()
    print(f"Layer innovations: {status['config']}")
    
    return layer


def demonstrate_mpq_progression():
    """Demonstrate Multi-stage Progressive Quantization."""
    print("\nDemonstrating MPQ progression...")
    
    # Create MPQ schedule
    mpq_config = bitnet_v3.create_mpq_schedule(
        total_epochs=20,
        num_stages=4,
        final_bits=1.58
    )
    
    # Create quantizer
    quantizer = bitnet_v3.MultiStageProgressiveQuantizer(mpq_config)
    
    # Simulate training progression
    test_weight = torch.randn(128, 256)
    
    for epoch in [1, 5, 10, 15, 20]:
        quantizer.set_epoch(epoch)
        quantized = quantizer(test_weight, is_weight=True)
        status = quantizer.get_current_status()
        
        print(f"Epoch {epoch}: Stage {status['stage_info']['stage']}, "
              f"Bits: {status['weight_quantization']['target_bits']}, "
              f"Progress: {status['stage_info']['progress']:.2f}")


def demonstrate_adaptive_hadamard():
    """Demonstrate Adaptive Hadamard Transform."""
    print("\nDemonstrating Adaptive Hadamard Transform...")
    
    # Create adaptive Hadamard transform
    aht = bitnet_v3.AdaptiveHadamardTransform(
        size=256,
        learnable_params=True,
        distribution_aware=True,
    )
    
    # Test transformation
    x = torch.randn(4, 64, 256)  # batch_size=4, seq_len=64, features=256
    
    # Forward pass
    aht.train()
    x_transformed = aht(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Transformed shape: {x_transformed.shape}")
    
    # Analyze adaptation
    analysis = aht.analyze_adaptation()
    print(f"Learned patterns: {analysis.get('learned_patterns', 'N/A')}")


def run_training_simulation():
    """Run a simple training simulation with BitNet v3."""
    print("\nRunning training simulation...")
    
    # Create small model
    config = bitnet_v3.BitNetV3Config(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=32,
    )
    
    model = bitnet_v3.BitNetV3ForCausalLM(config)
    
    # Create dummy dataset
    batch_size = 4
    seq_length = 16
    vocab_size = config.vocab_size
    
    # Generate random data
    input_ids = torch.randint(0, vocab_size, (batch_size * 10, seq_length))
    labels = torch.randint(0, vocab_size, (batch_size * 10, seq_length))
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    total_loss = 0
    
    print("Starting training simulation...")
    for epoch in range(3):
        model.set_epoch(epoch + 1)  # Set epoch for MPQ
        epoch_loss = 0
        
        for batch_idx, (batch_input_ids, batch_labels) in enumerate(dataloader):
            if batch_idx >= 5:  # Limit to 5 batches per epoch for demo
                break
                
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=batch_input_ids, labels=batch_labels)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / min(5, len(dataloader))
        total_loss += avg_loss
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
    
    print(f"Training completed. Final average loss: {total_loss / 3:.4f}")
    
    # Get BitNet status
    status = model.get_bitnet_status()
    print(f"Model has {status['num_layers']} BitNet v3 layers")


def demonstrate_generation():
    """Demonstrate text generation with BitNet v3."""
    print("\nDemonstrating text generation...")
    
    # Create small model for generation
    config = bitnet_v3.BitNetV3Config(
        vocab_size=50,
        hidden_size=128,
        num_layers=4,
        num_heads=8,
        max_position_embeddings=64,
    )
    
    model = bitnet_v3.BitNetV3ForCausalLM(config)
    model.eval()
    
    # Generate text
    input_ids = torch.randint(0, config.vocab_size, (1, 5))  # Start with 5 tokens
    
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Generate continuation
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=15,
            temperature=1.0,
            do_sample=True,
            top_k=10,
        )
    
    print(f"Generated tokens: {generated[0].tolist()}")
    print("Generation completed successfully!")


def main():
    """Run all demonstrations."""
    print("BitNet v3 Basic Usage Examples")
    print("=" * 50)
    
    try:
        # 1. Create model
        model = create_simple_model()
        
        # 2. Demonstrate BitLinear
        layer = demonstrate_bitlinear_layer()
        
        # 3. Demonstrate MPQ
        demonstrate_mpq_progression()
        
        # 4. Demonstrate Adaptive Hadamard
        demonstrate_adaptive_hadamard()
        
        # 5. Run training simulation
        run_training_simulation()
        
        # 6. Demonstrate generation
        demonstrate_generation()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("BitNet v3 package is working correctly.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()