import torch
import torch.nn as nn
import unittest
import math # For isclose

# Assuming bitnet_v3 is in PYTHONPATH or installed
from bitnet_v3.core.quantization import (
    QuantizationConfig,
    BitNetQuantizer,
    round_ste, # Will also test ScaledRoundSTE through this
    compute_quantization_error,
    pack_ternary_weights,
    unpack_ternary_weights,
    absmean_quantization, # For testing scale_multiplier and LSQ-like step
    absmax_quantization
)
# from bitnet_v3.modules.bitlinear import BitLinear # To test integration of BitNetQuantizer - removed for focus

class TestQuantizationCore(unittest.TestCase):

    def test_round_ste_gradient_scaling(self):
        # Test ScaledRoundSTE backward pass
        x = torch.tensor([0.2, 0.7, 1.3, 1.8], requires_grad=True)
        grad_scale_factor = 0.5

        # Test via round_ste function
        y = round_ste(x, grad_scale_factor)
        y.sum().backward()

        expected_grad = torch.ones_like(x) * grad_scale_factor
        self.assertTrue(torch.allclose(x.grad, expected_grad), f"Expected grad {expected_grad}, got {x.grad}")

        # Test default (factor=1.0)
        x.grad = None
        y_default = round_ste(x, 1.0)
        y_default.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)), f"Expected grad {torch.ones_like(x)}, got {x.grad}")

    def test_compute_quantization_error_methods(self):
        original = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        quantized_l2 = torch.tensor([[1.1, 2.1, 3.1], [-0.9, -1.9, -2.9]])
        quantized_l1 = torch.tensor([[1.0, 2.5, 3.0], [-1.5, -2.0, -2.5]]) # larger L1 errors in places

        error_l2 = compute_quantization_error(original, quantized_l2, method="l2")
        self.assertGreater(error_l2.item(), 0) # Basic check

        error_l1 = compute_quantization_error(original, quantized_l1, method="l1")
        self.assertGreater(error_l1.item(), 0)

        # Test moments
        # For moments, ensure means/vars are different enough to produce non-zero error
        quantized_moments = quantized_l1 * 1.5 + 0.5
        error_moments = compute_quantization_error(original, quantized_moments, method="moments", moment_weights=(1.0, 0.5))
        self.assertGreater(error_moments.item(), 0)

        with self.assertRaises(ValueError):
            compute_quantization_error(original, quantized_l2, method="unknown")

    def test_bit_packing_unpacking_roundtrip(self):
        # Test case 1: simple {-1, 0, 1}
        weights1 = torch.tensor([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]], dtype=torch.float32)
        true_pos1, true_neg1 = 1.0, -1.0
        packed1 = pack_ternary_weights(weights1, true_pos1, true_neg1)
        unpacked1 = unpack_ternary_weights(packed1, weights1.shape, true_pos1, true_neg1)
        self.assertTrue(torch.equal(weights1, unpacked1), "Roundtrip failed for simple {-1,0,1} case")

        # Test case 2: scaled values
        v_pos, v_neg = 0.75, -0.65
        weights2 = torch.tensor([[v_pos, 0.0, v_neg], [0.0, v_pos, v_pos]], dtype=torch.float32)
        packed2 = pack_ternary_weights(weights2, v_pos, v_neg)
        unpacked2 = unpack_ternary_weights(packed2, weights2.shape, v_pos, v_neg)
        self.assertTrue(torch.allclose(weights2, unpacked2), "Roundtrip failed for scaled values")

        # Test case 3: all zeros
        weights3 = torch.zeros((2, 4), dtype=torch.float32)
        packed3 = pack_ternary_weights(weights3, 1.0, -1.0) # true_pos/neg don't matter much if all zero
        unpacked3 = unpack_ternary_weights(packed3, weights3.shape, 1.0, -1.0)
        self.assertTrue(torch.equal(weights3, unpacked3), "Roundtrip failed for all zeros")

        # Test case 4: requires padding
        weights4 = torch.tensor([1.0, 0.0, -1.0, 1.0, 0.0], dtype=torch.float32) # 5 elements
        packed4 = pack_ternary_weights(weights4, 1.0, -1.0)
        unpacked4 = unpack_ternary_weights(packed4, weights4.shape, 1.0, -1.0)
        self.assertTrue(torch.equal(weights4, unpacked4), "Roundtrip failed for tensor requiring padding")

    def test_scale_learnable_multiplier_effect(self):
        # Test through absmean_quantization
        x = torch.randn(5, 10)
        # Using activation_bits for multi-bit path, method for absmean

        # No multiplier
        q_no_mult, scale_no_mult = absmean_quantization(x, bits=8, method="absmean") # bits > 1.58

        # With multiplier
        multiplier_val = 0.5
        multiplier_param = torch.tensor(multiplier_val)
        q_with_mult, scale_with_mult = absmean_quantization(x, bits=8, method="absmean", scale_multiplier=multiplier_param)

        self.assertTrue(torch.allclose(scale_with_mult, scale_no_mult * multiplier_val), "Scale multiplier did not apply correctly")
        # Quantized values should differ if scale differs significantly
        self.assertFalse(torch.allclose(q_no_mult, q_with_mult) or x.abs().max() < 1e-6 , "Quantized values unexpectedly close or input too small.")

    def test_learnable_parameters_in_quantizer_components(self):
        config = QuantizationConfig(
            weight_bits=1.58,
            activation_bits=4,
            enable_scale_learnable_multiplier=True, # For activations & weights
            round_ste_grad_scale_factor=0.9,
            ternary_threshold=0.4,
            ternary_neg_val_init=0.9,
            ternary_pos_val_init=1.1,
            activation_learnable_step_init=1.2
        )

        quantizer = BitNetQuantizer(config)

        # Check weight quantizer's learnable parameters
        self.assertTrue(isinstance(quantizer.weight_quantizer.learnable_ternary_threshold, nn.Parameter))
        self.assertTrue(isinstance(quantizer.weight_quantizer.quant_neg_val, nn.Parameter))
        self.assertTrue(isinstance(quantizer.weight_quantizer.quant_pos_val, nn.Parameter))
        self.assertTrue(isinstance(quantizer.weight_quantizer.scale_multiplier, nn.Parameter)) # from enable_scale_learnable_multiplier

        # Check activation quantizer's learnable parameters
        self.assertTrue(isinstance(quantizer.activation_quantizer.scale_multiplier, nn.Parameter))
        self.assertTrue(isinstance(quantizer.activation_quantizer.activation_learnable_step, nn.Parameter))

        # Test gradient flow for one parameter as an example: learnable_ternary_threshold
        param_to_test = quantizer.weight_quantizer.learnable_ternary_threshold
        initial_val = param_to_test.clone().detach()

        optimizer = torch.optim.SGD([param_to_test], lr=0.1)
        optimizer.zero_grad()

        weight_data = torch.randn(8, 4)
        quantized_w = quantizer.quantize_weights(weight_data)

        loss = (quantized_w - (weight_data + 0.1)).pow(2).sum()
        loss.backward()

        self.assertIsNotNone(param_to_test.grad, "Gradient for learnable_ternary_threshold is None")
        self.assertNotEqual(param_to_test.grad.abs().sum().item(), 0.0, "Gradient for learnable_ternary_threshold is zero")

        optimizer.step()
        self.assertNotEqual(initial_val.item(), param_to_test.item(), "learnable_ternary_threshold did not update.")

if __name__ == '__main__':
    # This allows running the tests from the command line
    # Create a 'tests' directory if it doesn't exist at the root of your project
    # and place this file (test_quantization.py) in it.
    # Then run: python -m unittest tests.test_quantization
    unittest.main()
