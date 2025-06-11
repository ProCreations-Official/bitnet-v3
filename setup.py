#!/usr/bin/env python3
"""
BitNet v3: Ultra-Low Quality Loss 1-bit LLMs Through Multi-Stage Progressive Quantization and Adaptive Hadamard Transform

Setup script for the BitNet v3 package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read the requirements file
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="bitnet-v3",
    version="1.0.0",
    author="ProCreations",
    author_email="procreations@example.com",
    description="BitNet v3: Ultra-Low Quality Loss 1-bit LLMs Through Multi-Stage Progressive Quantization and Adaptive Hadamard Transform",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ProCreations/bitnet-v3",
    project_urls={
        "Bug Tracker": "https://github.com/ProCreations/bitnet-v3/issues",
        "Documentation": "https://github.com/ProCreations/bitnet-v3/blob/main/README.md",
        "Source Code": "https://github.com/ProCreations/bitnet-v3",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "examples": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "tqdm>=4.60",
            "datasets>=1.8",
            "transformers>=4.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "bitnet-v3-train=bitnet_v3.cli:train_cli",
            "bitnet-v3-infer=bitnet_v3.cli:infer_cli",
        ],
    },
    keywords=[
        "bitnet",
        "quantization", 
        "1-bit",
        "llm",
        "transformer",
        "hadamard",
        "knowledge-distillation",
        "neural-networks",
        "deep-learning",
        "pytorch",
    ],
    package_data={
        "bitnet_v3": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)