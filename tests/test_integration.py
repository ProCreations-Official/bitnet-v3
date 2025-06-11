"""
Integration tests for BitNet v3 package.

Tests all major components working together.
"""

import torch
import torch.nn as nn
import tempfile
import os

# Import all components
import bitnet_v3
from bitnet_v3.core.quantization import quantize_weights_158, quantize_activations
from bitnet_v3.core.hadamard import hadamard_transform, create_hadamard_matrix
from bitnet_v3.modules.mpq import MultiStageProgressiveQuantizer, create_mpq_schedule
from bitnet_v3.modules.aht_lp import AdaptiveHadamardTransform
from bitnet_v3.modules.gakd import GradientAwareKnowledgeDistillation
from bitnet_v3.modules.dr_qap import DynamicRegularizationQAP
from bitnet_v3.modules.este_m import EnhancedSTEMomentum
from bitnet_v3.modules.bitlinear import EnhancedHBitLinear, create_bitlinear_layer
from bitnet_v3.models.bitnet_v3 import BitNetV3Model, BitNetV3Config, BitNetV3ForCausalLM


class TestCoreComponents:
    """Test core quantization and Hadamard components."""
    
    def test_quantization_functions(self):
        """Test quantization functions work correctly."""
        # Test weight quantization
        weight = torch.randn(32, 64)
        quantized_weight = quantize_weights_158(weight)
        
        # Check output is ternary
        unique_values = torch.unique(quantized_weight)
        assert len(unique_values) <= 3, "Weight quantization should produce ternary values"
        assert torch.all(torch.isin(unique_values, torch.tensor([-1.0, 0.0, 1.0]))), "Invalid quantized values"
        
        # Test activation quantization
        activation = torch.randn(8, 32, 128)
        quantized_activation = quantize_activations(activation, bits=4)
        
        assert quantized_activation.shape == activation.shape, "Shape should be preserved"
        
    def test_hadamard_transform(self):
        """Test Hadamard transform functions."""
        # Test with power-of-2 size
        x = torch.randn(4, 16, 64)  # 64 is power of 2
        x_transformed = hadamard_transform(x)
        
        assert x_transformed.shape == x.shape, "Shape should be preserved"
        
        # Test Hadamard matrix creation
        H = create_hadamard_matrix(8)
        assert H.shape == (8, 8), "Hadamard matrix shape incorrect"
        
        # Test orthogonality (H @ H.T should be identity * n)
        HHT = H @ H.t()
        expected = torch.eye(8) * 8
        assert torch.allclose(HHT, expected, atol=1e-5), "Hadamard matrix not orthogonal"


class TestModules:
    """Test individual innovation modules."""
    
    def test_mpq_module(self):
        """Test Multi-stage Progressive Quantization."""
        config = create_mpq_schedule(total_epochs=20, num_stages=4)
        quantizer = MultiStageProgressiveQuantizer(config)
        
        # Test with different epochs
        weight = torch.randn(32, 64)
        
        for epoch in [1, 10, 20]:
            quantizer.set_epoch(epoch)
            quantized = quantizer(weight, is_weight=True)
            assert quantized.shape == weight.shape, "Shape preservation failed"
            
            status = quantizer.get_current_status()
            assert 'stage_info' in status, "Status should contain stage info"
    
    def test_adaptive_hadamard(self):
        """Test Adaptive Hadamard Transform with Learnable Parameters."""
        aht = AdaptiveHadamardTransform(
            size=128,
            learnable_params=True,
            distribution_aware=True
        )
        
        x = torch.randn(2, 32, 128)
        
        # Test forward pass
        aht.train()
        x_transformed = aht(x)
        
        assert x_transformed.shape == x.shape, "Shape should be preserved"
        
        # Test learnable parameters exist
        assert hasattr(aht, 'learnable_params_module'), "Should have learnable parameters"
        
        # Test analysis
        analysis = aht.analyze_adaptation()
        assert 'size' in analysis, "Analysis should contain size info"
    
    def test_gakd_module(self):
        """Test Gradient-Aware Knowledge Distillation."""
        gakd = GradientAwareKnowledgeDistillation(
            alpha=0.7, beta=0.2, gamma=0.1
        )
        
        # Test loss computation
        student_outputs = torch.randn(4, 10, 100)  # batch, seq, vocab
        teacher_outputs = torch.randn(4, 10, 100)
        
        losses = gakd.gakd_loss(student_outputs, teacher_outputs)
        
        assert 'total_loss' in losses, "Should compute total loss"
        assert 'kl_loss' in losses, "Should compute KL loss"
        assert losses['total_loss'].requires_grad, "Loss should require gradients"
    
    def test_dynamic_regularization(self):
        """Test Dynamic Regularization with QAP."""
        dr_qap = DynamicRegularizationQAP(
            initial_lambda=0.1,
            warmup_steps=5
        )
        
        # Create dummy model
        model = nn.Linear(64, 32)
        
        # Test penalty computation
        penalty = dr_qap(model)
        
        assert isinstance(penalty, torch.Tensor), "Should return tensor"
        assert penalty.numel() == 1, "Should be scalar"
    
    def test_enhanced_ste(self):
        """Test Enhanced STE with Momentum."""
        este_m = EnhancedSTEMomentum(
            momentum=0.9,
            power=0.5,
            enable_enhanced_ste=True
        )
        
        # Test gradient enhancement
        x = torch.randn(16, 32, requires_grad=True)
        
        def dummy_quantize(tensor):
            return tensor.round()
        
        # Forward pass
        output = este_m(x, dummy_quantize)
        
        assert output.shape == x.shape, "Shape should be preserved"
        
        # Test metrics
        metrics = este_m.get_performance_metrics()
        assert 'step_count' in metrics, "Should track steps"


class TestBitLinear:
    """Test Enhanced H-BitLinear layer."""
    
    def test_bitlinear_creation(self):
        """Test BitLinear layer creation."""
        layer = create_bitlinear_layer(
            in_features=128,
            out_features=256,
            enable_all_innovations=True
        )
        
        assert isinstance(layer, EnhancedHBitLinear), "Should create BitLinear layer"
        assert layer.in_features == 128, "Input features incorrect"
        assert layer.out_features == 256, "Output features incorrect"
    
    def test_bitlinear_forward(self):
        """Test BitLinear forward pass."""
        layer = create_bitlinear_layer(64, 128, enable_all_innovations=True)
        
        x = torch.randn(2, 32, 64)
        
        # Test forward pass
        output = layer(x, epoch=1)
        
        assert output.shape == (2, 32, 128), "Output shape incorrect"
        
        # Test quantization status
        status = layer.get_quantization_status()
        assert 'config' in status, "Should have config in status"
    
    def test_bitlinear_epoch_setting(self):
        """Test epoch setting for MPQ."""
        layer = create_bitlinear_layer(32, 64, enable_all_innovations=True)
        
        # Test epoch setting
        layer.set_epoch(10)
        
        # Forward pass should work with epoch
        x = torch.randn(4, 16, 32)
        output = layer(x, epoch=10)
        
        assert output.shape == (4, 16, 64), "Forward with epoch failed"


class TestModels:
    """Test complete BitNet v3 models."""
    
    def test_model_creation(self):
        """Test BitNet v3 model creation."""
        config = BitNetV3Config(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_position_embeddings=32,
        )
        
        model = BitNetV3Model(config)
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0, "Model should have parameters"
        
        # Test configuration
        assert model.config.vocab_size == 100, "Config not preserved"
    
    def test_causal_lm_model(self):
        """Test BitNet v3 for Causal LM."""
        config = BitNetV3Config(
            vocab_size=50,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_position_embeddings=16,
        )
        
        model = BitNetV3ForCausalLM(config)
        
        # Test forward pass
        input_ids = torch.randint(0, 50, (2, 8))
        
        outputs = model(input_ids)
        
        # Check outputs
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs[1]
        
        assert logits.shape == (2, 8, 50), "Logits shape incorrect"
    
    def test_model_with_loss(self):
        """Test model forward pass with loss computation."""
        config = BitNetV3Config(
            vocab_size=30,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            max_position_embeddings=16,
        )
        
        model = BitNetV3ForCausalLM(config)
        
        # Test with labels
        input_ids = torch.randint(0, 30, (2, 8))
        labels = torch.randint(0, 30, (2, 8))
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        # Check loss
        if isinstance(outputs, dict):
            loss = outputs['loss']
        else:
            loss = outputs[0]
        
        assert loss is not None, "Loss should be computed"
        assert loss.requires_grad, "Loss should require gradients"
    
    def test_model_epoch_setting(self):
        """Test epoch setting for model MPQ."""
        config = BitNetV3Config(
            vocab_size=20,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            max_position_embeddings=8,
        )
        
        model = BitNetV3ForCausalLM(config)
        
        # Test epoch setting
        model.set_epoch(5)
        assert model.current_epoch == 5, "Epoch not set correctly"
        
        # Test forward pass with epoch
        input_ids = torch.randint(0, 20, (1, 4))
        outputs = model(input_ids)
        
        assert outputs is not None, "Forward pass with epoch failed"


class TestGeneration:
    """Test text generation capabilities."""
    
    def test_basic_generation(self):
        """Test basic text generation."""
        config = BitNetV3Config(
            vocab_size=25,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            max_position_embeddings=16,
        )
        
        model = BitNetV3ForCausalLM(config)
        model.eval()
        
        # Test generation
        input_ids = torch.randint(0, 25, (1, 3))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_length=8,
                temperature=1.0,
                do_sample=False,  # Greedy for deterministic test
            )
        
        assert generated.shape[0] == 1, "Batch size should be preserved"
        assert generated.shape[1] <= 8, "Should not exceed max_length"
        assert generated.shape[1] >= input_ids.shape[1], "Should not be shorter than input"


class TestSaveLoad:
    """Test model saving and loading."""
    
    def test_config_save_load(self):
        """Test configuration saving and loading."""
        config = BitNetV3Config(
            vocab_size=100,
            hidden_size=128,
            num_layers=4,
            num_heads=8,
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save configuration
            config.save_pretrained(temp_dir)
            
            # Check file exists
            config_file = os.path.join(temp_dir, "config.json")
            assert os.path.exists(config_file), "Config file not created"
            
            # Load configuration
            loaded_config = BitNetV3Config.from_pretrained(temp_dir)
            
            # Check values
            assert loaded_config.vocab_size == config.vocab_size, "Vocab size not preserved"
            assert loaded_config.hidden_size == config.hidden_size, "Hidden size not preserved"
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        config = BitNetV3Config(
            vocab_size=50,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
        )
        
        original_model = BitNetV3Model(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            original_model.save_pretrained(temp_dir)
            
            # Check files exist
            assert os.path.exists(os.path.join(temp_dir, "config.json")), "Config not saved"
            assert os.path.exists(os.path.join(temp_dir, "pytorch_model.bin")), "Model not saved"
            
            # Load model
            loaded_model = BitNetV3Model.from_pretrained(temp_dir)
            
            # Check configuration
            assert loaded_model.config.vocab_size == config.vocab_size, "Config not loaded correctly"


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_training_pipeline(self):
        """Test a complete training pipeline."""
        # Create small model
        config = BitNetV3Config(
            vocab_size=30,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            max_position_embeddings=8,
            enable_mpq=True,
            enable_adaptive_hadamard=True,
            enable_enhanced_ste=True,
        )
        
        model = BitNetV3ForCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Simulate training for a few steps
        model.train()
        
        for epoch in range(3):
            model.set_epoch(epoch + 1)
            
            # Create batch
            input_ids = torch.randint(0, 30, (2, 6))
            labels = torch.randint(0, 30, (2, 6))
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        
        # Test inference
        model.eval()
        with torch.no_grad():
            test_input = torch.randint(0, 30, (1, 3))
            outputs = model(test_input)
            assert outputs is not None, "Inference failed"
    
    def test_bitnet_status_reporting(self):
        """Test BitNet v3 status reporting."""
        config = BitNetV3Config(
            vocab_size=20,
            hidden_size=32,
            num_layers=2,
            num_heads=2,
            enable_bitnet_innovations=True,
        )
        
        model = BitNetV3ForCausalLM(config)
        
        # Get status
        status = model.get_bitnet_status()
        
        assert 'num_layers' in status, "Should report number of layers"
        assert status['num_layers'] == 2, "Should report correct layer count"
        assert 'layers' in status, "Should report layer details"


def test_package_import():
    """Test that the package imports correctly."""
    # Test main package import
    assert hasattr(bitnet_v3, 'BitNetV3Model'), "Main model class not available"
    assert hasattr(bitnet_v3, 'BitNetV3Config'), "Config class not available"
    assert hasattr(bitnet_v3, 'create_model'), "Convenience function not available"
    
    # Test core functions
    assert hasattr(bitnet_v3, 'quantize_weights_158'), "Quantization function not available"
    assert hasattr(bitnet_v3, 'hadamard_transform'), "Hadamard function not available"
    
    # Test module classes
    assert hasattr(bitnet_v3, 'EnhancedHBitLinear'), "BitLinear class not available"
    assert hasattr(bitnet_v3, 'MultiStageProgressiveQuantizer'), "MPQ class not available"


def run_all_tests():
    """Run all tests manually (for environments without pytest)."""
    print("Running BitNet v3 Integration Tests...")
    print("=" * 50)
    
    test_classes = [
        TestCoreComponents,
        TestModules, 
        TestBitLinear,
        TestModels,
        TestGeneration,
        TestSaveLoad,
        TestIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        test_instance = test_class()
        
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
    
    # Run standalone tests
    try:
        test_package_import()
        print(f"  ✓ test_package_import")
        total_tests += 1
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ test_package_import: {e}")
        total_tests += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! BitNet v3 package is working correctly.")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} tests failed.")
        return False


if __name__ == "__main__":
    # Run tests directly
    success = run_all_tests()
    exit(0 if success else 1)