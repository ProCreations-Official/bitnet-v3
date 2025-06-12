# BitNet v3: Ultra-Low Quality Loss 1-bit LLMs

[![PyPI version](https://badge.fury.io/py/bitnet-v3.svg)](https://badge.fury.io/py/bitnet-v3)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive PyTorch implementation of **BitNet v3**, a novel framework for training 1-bit Large Language Models (LLMs) that significantly reduces quality loss while maintaining computational efficiency benefits of extreme quantization.

## 🚀 Key Features

BitNet v3 introduces **five key innovations** for ultra-low quality loss 1-bit LLMs:

1. **🔄 Multi-stage Progressive Quantization (MPQ)** - Gradually reduces bit-width during training
2. **🧮 Adaptive Hadamard Transform with Learnable Parameters (AHT-LP)** - Dynamically adjusts to activation distributions
3. **🎓 Gradient-Aware Knowledge Distillation (GAKD)** - Preserves critical gradient information during quantization
4. **⚖️ Dynamic Regularization with Quantization-Aware Penalties (DR-QAP)** - Stabilizes training with adaptive penalties
5. **💫 Enhanced Straight-Through Estimator with Momentum (ESTE-M)** - Improves gradient approximation

## 🔬 Research Status

This implementation provides the framework for training 1-bit LLMs with the potential for significant quality improvements over existing methods. **Performance evaluation is ongoing** - we're actively seeking contributors to help with testing, benchmarking, and validation across different model sizes and datasets.

---

> **🚨 CONTRIBUTORS WANTED!** 🚨  
> **Help us validate BitNet v3!** We need researchers and engineers to test performance, optimize code, and validate results. All skill levels welcome - from bug reports to research contributions. [Jump to Contributing section](#-contributing---we-need-your-help) or [start a discussion](https://github.com/ProCreations-Official/bitnet-v3/discussions)!

---

## 🛠️ Installation

### From PyPI (Recommended)
```bash
pip install bitnet-v3
```

### From Source
```bash
git clone https://github.com/ProCreations-Official/bitnet-v3.git
cd bitnet-v3
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/ProCreations-Official/bitnet-v3.git
cd bitnet-v3
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Simple Usage

```python
import bitnet_v3

# Create a BitNet v3 model
model = bitnet_v3.create_model(
    vocab_size=32000,
    hidden_size=2048,
    num_layers=24,
    num_heads=32,
)

# Create trainer with MPQ schedule
trainer = bitnet_v3.create_trainer(
    model,
    learning_rate=3e-4,
    batch_size=256,
    enable_mpq=True,
    enable_gakd=True,
)

# Train the model
trainer.train(train_dataloader)
```

### Advanced Usage with All Features

```python
import torch
import bitnet_v3

# Configure model with all innovations
config = bitnet_v3.BitNetV3Config(
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    # MPQ configuration
    mpq_stages=[
        {"epochs": 20, "bits": 8},
        {"epochs": 20, "bits": 4}, 
        {"epochs": 15, "bits": 2},
        {"epochs": 15, "bits": 1.58},
    ],
    # AHT-LP configuration
    adaptive_hadamard=True,
    hadamard_learnable_scale=True,
    # GAKD configuration
    knowledge_distillation=True,
    gakd_alpha=0.7,
    gakd_beta=0.2,
    gakd_gamma=0.1,
    # DR-QAP configuration
    dynamic_regularization=True,
    qap_initial_lambda=0.1,
    # ESTE-M configuration
    enhanced_ste=True,
    ste_momentum=0.9,
)

# Create model and trainer
model = bitnet_v3.BitNetV3Model(config)
trainer = bitnet_v3.BitNetV3Trainer(model, config)

# Load teacher model for knowledge distillation
teacher_model = torch.load("teacher_model.pth")
trainer.set_teacher_model(teacher_model)

# Train with all features
trainer.train(
    train_dataloader,
    val_dataloader,
    num_epochs=70,
    save_every=5,
    eval_every=1,
)
```

## 🏗️ Architecture Overview

### Core Components

- **`bitnet_v3.core`** - Core quantization functions and utilities
- **`bitnet_v3.modules`** - Individual innovation modules (MPQ, AHT-LP, GAKD, etc.)
- **`bitnet_v3.models`** - Complete BitNet v3 model implementations
- **`bitnet_v3.training`** - Training pipeline and utilities
- **`bitnet_v3.utils`** - Configuration, logging, and metrics

### Key Modules

```python
# Enhanced H-BitLinear with all innovations
linear_layer = bitnet_v3.EnhancedHBitLinear(
    in_features=2048,
    out_features=2048,
    bias=False,
    adaptive_hadamard=True,
    progressive_quantization=True,
)

# Multi-stage Progressive Quantizer
mpq = bitnet_v3.MultiStageProgressiveQuantizer(
    stages=[8, 4, 2, 1.58],
    stage_epochs=[20, 20, 15, 15],
)

# Adaptive Hadamard Transform
aht = bitnet_v3.AdaptiveHadamardTransform(
    size=2048,
    learnable_params=True,
)

# Gradient-Aware Knowledge Distillation
gakd = bitnet_v3.GradientAwareKnowledgeDistillation(
    alpha=0.7,  # KL divergence weight
    beta=0.2,   # Gradient alignment weight  
    gamma=0.1,  # Feature alignment weight
)
```

## 📚 Detailed Documentation

### Multi-Stage Progressive Quantization (MPQ)

MPQ gradually reduces bit-width during training, allowing models to adapt smoothly:

```python
# Configure MPQ stages
mpq_config = {
    "stages": [
        {"start_epoch": 1, "end_epoch": 20, "bits": 8},
        {"start_epoch": 21, "end_epoch": 40, "bits": 4},
        {"start_epoch": 41, "end_epoch": 55, "bits": 2},
        {"start_epoch": 56, "end_epoch": 70, "bits": 1.58},
    ],
    "temperature_schedule": "linear",  # or "cosine"
}

scheduler = bitnet_v3.MPQScheduler(**mpq_config)
```

### Adaptive Hadamard Transform (AHT-LP)

Enhanced Hadamard transformation with learnable parameters:

```python
# Standard Hadamard transform
x_transformed = bitnet_v3.hadamard_transform(x)

# Adaptive Hadamard with learnable parameters
aht = bitnet_v3.AdaptiveHadamardTransform(
    size=x.size(-1),
    learnable_scale=True,
    learnable_shift=True,
)
x_adaptive = aht(x)
```

### Gradient-Aware Knowledge Distillation (GAKD)

Preserves gradient information during distillation:

```python
# Set up GAKD
gakd_loss = bitnet_v3.GradientAwareKnowledgeDistillation(
    alpha=0.7,  # Output distribution weight
    beta=0.2,   # Gradient alignment weight
    gamma=0.1,  # Feature alignment weight
)

# Compute distillation loss
loss = gakd_loss(
    student_outputs,
    teacher_outputs,
    student_features,
    teacher_features,
    student_gradients,
    teacher_gradients,
)
```

## 🧪 Examples

### Training from Scratch

```python
import bitnet_v3
from torch.utils.data import DataLoader

# Load your dataset
train_dataset = YourDataset("train")
train_loader = DataLoader(train_dataset, batch_size=256)

# Create model with default config
model = bitnet_v3.create_model(
    vocab_size=len(tokenizer),
    hidden_size=2048,
    num_layers=24,
)

# Train with MPQ
trainer = bitnet_v3.create_trainer(model)
trainer.train(train_loader, num_epochs=70)
```

### Fine-tuning Pre-trained Model

```python
# Load pre-trained model
model = bitnet_v3.BitNetV3Model.from_pretrained("path/to/model")

# Convert to BitNet v3 with progressive quantization
bitnet_model = bitnet_v3.convert_to_bitnet_v3(
    model,
    enable_all_features=True,
)

# Fine-tune with knowledge distillation
trainer = bitnet_v3.create_trainer(bitnet_model)
trainer.set_teacher_model(model)  # Use original as teacher
trainer.train(fine_tune_loader, num_epochs=20)
```

### Inference

```python
# Load trained BitNet v3 model
model = bitnet_v3.BitNetV3Model.from_pretrained("path/to/bitnet_v3_model")

# Generate text
output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,
    do_sample=True,
)
```

## 🔬 Research Paper Implementation

This implementation includes all techniques from the original BitNet v3 research paper:

### Quantization Functions
- Ternary weight quantization: `{-1, 0, 1}`
- 4-bit activation quantization with Hadamard transform
- AbsMean and AbsMax quantization schemes

### Training Innovations
- Progressive bit-width reduction schedule
- Temperature-based quantization transitions
- Gradient-aware loss computation
- Dynamic regularization with layer sensitivity

### Mathematical Formulations
All key equations from the paper are implemented:

```python
# Temperature-based transition (Equation 1)
Q_t(x) = σ(β_t) * Q_b_t(x) + (1 - σ(β_t)) * Q_b_{t-1}(x)

# Adaptive Hadamard transform (Equation 2)  
H_adaptive(x) = γ ⊙ (H_m · x) + β

# GAKD loss (Equation 3)
L_GAKD = α*L_KL + β*L_grad + γ*L_feature

# Dynamic regularization (Equation 4)
R_QAP = λ(t) * Σ ω_i ||W_i - Q(W_i)||²
```

## 📈 Evaluation and Metrics

Built-in evaluation tools for comprehensive analysis:

```python
# Compute perplexity
ppl = bitnet_v3.compute_perplexity(model, test_loader)

# Efficiency metrics (theoretical analysis)
metrics = bitnet_v3.compute_efficiency_metrics(
    bitnet_model, 
    baseline_model,
    test_input,
)
print(f"Expected speedup: {metrics['speedup']:.1f}x")
print(f"Expected memory reduction: {metrics['memory_reduction']:.1f}%")

# Downstream task evaluation framework
results = bitnet_v3.evaluate_downstream_tasks(
    model,
    tasks=["hellaswag", "mmlu", "truthfulqa"],
)
```

**Note**: Performance validation is ongoing. We encourage the community to help benchmark BitNet v3 across different tasks and model sizes.

## 🛡️ Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_modules/test_mpq.py
pytest tests/test_modules/test_gakd.py

# Run with coverage
pytest --cov=bitnet_v3 tests/
```

## 🤝 Contributing - We Need Your Help!

**BitNet v3 is an active research project and we're actively seeking contributors!** Whether you're a researcher, engineer, or enthusiast, there are many ways to contribute to advancing 1-bit LLM technology.

### 🚨 High Priority Contributions Needed

- **🧪 Performance Benchmarking**: Help us validate BitNet v3 across different model sizes (1B, 3B, 7B+)
- **📊 Dataset Testing**: Test on various datasets (language modeling, downstream tasks, multilingual)
- **⚡ Optimization**: CUDA kernels, memory optimizations, training speed improvements
- **🔧 Integration**: HuggingFace Transformers integration, ONNX export, deployment tools
- **📝 Documentation**: Tutorials, guides, and improved examples
- **🐛 Bug Reports**: Help us identify and fix issues in the codebase

### 🎯 Research Opportunities

- Compare against other quantization methods (GPTQ, AWQ, etc.)
- Explore different MPQ schedules and temperature functions
- Investigate GAKD effectiveness across model architectures
- Test on specialized domains (code, math, science)
- Efficiency analysis on different hardware (GPUs, edge devices)

### 💡 Easy Ways to Get Started

1. **Run the examples** and report any issues
2. **Test installation** on different systems (Windows, Mac, Linux)
3. **Improve documentation** - add docstrings, fix typos, clarify explanations
4. **Add unit tests** for untested modules
5. **Create tutorials** for specific use cases

### 🛠 Development Setup

```bash
# Fork the repository on GitHub first!
git clone https://github.com/ProCreations-Official/bitnet-v3.git
cd bitnet-v3

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests to ensure everything works
pytest tests/
```

### 📋 Contribution Guidelines

- **All skill levels welcome** - from typo fixes to major algorithmic improvements
- **Research-first approach** - we prioritize correctness and reproducibility
- **Open communication** - discuss ideas in GitHub Issues before major changes
- **Documentation required** - all new features need documentation and examples
- **Testing encouraged** - add tests for new functionality when possible

### 🌟 Recognition

Contributors will be:
- Added to the contributors list in the README
- Acknowledged in any resulting research papers
- Invited to collaborate on follow-up research

### 📞 Get in Touch

- 💬 **Start a Discussion**: [GitHub Discussions](https://github.com/ProCreations-Official/bitnet-v3/discussions) for questions and ideas
- 🐛 **Report Issues**: [GitHub Issues](https://github.com/ProCreations-Official/bitnet-v3/issues) for bugs and feature requests

**Every contribution matters** - from fixing a typo to implementing a new feature. Join us in making 1-bit LLMs a reality! 🚀

## 📄 Citation

If you use BitNet v3 in your research, please cite this repository:

```bibtex
@software{bitnet_v3_2024,
  title={BitNet v3: Ultra-Low Quality Loss 1-bit LLMs Through Multi-Stage Progressive Quantization and Adaptive Hadamard Transform},
  author={ProCreations},
  url={https://github.com/ProCreations-Official/bitnet-v3},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the foundation of BitNet and BitNet b1.58 from Microsoft Research
- Inspired by advances in quantization-aware training and knowledge distillation
- Thanks to the PyTorch team for the excellent deep learning framework

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/ProCreations-Official/bitnet-v3/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ProCreations-Official/bitnet-v3/discussions)

---

**BitNet v3** - Bringing 1-bit LLMs closer to practical deployment! 🚀
