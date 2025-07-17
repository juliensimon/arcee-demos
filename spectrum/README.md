# Spectrum: Model Quantization and Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)

Configuration files and examples for model quantization and optimization.

## Overview

This directory contains configuration files, examples, and utilities for model quantization and optimization using Arcee's Spectrum framework. Spectrum provides tools for reducing model size and improving inference efficiency while maintaining performance.

## Contents

- `fft-8b.yaml`: Configuration for 8-bit Fast Fourier Transform quantization
- `lora-8b.yml`: Configuration for 8-bit LoRA (Low-Rank Adaptation)
- `qlora-8b.yml`: Configuration for 8-bit QLoRA (Quantized Low-Rank Adaptation)
- `spectrum-25.yml`: Spectrum configuration with 25% unfrozen parameters
- `spectrum-50.yml`: Spectrum configuration with 50% unfrozen parameters
- `svd-example.py`: Example script for Singular Value Decomposition
- `snr_results_*.json/yaml`: Signal-to-Noise Ratio results for various model configurations

## Techniques

### Quantization Methods
- **FFT-Based Quantization**: Frequency domain quantization
- **QLoRA**: Quantized Low-Rank Adaptation for efficient fine-tuning
- **Singular Value Decomposition**: Matrix factorization for model compression

### Parameter Efficiency
- **Partial Parameter Unfreezing**: Selectively training portions of the model
- **Signal-to-Noise Ratio Analysis**: Identifying optimal quantization levels

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install torch transformers peft
   ```

2. Run the SVD example:
   ```bash
   python svd-example.py
   ```

3. Use the configuration files with Arcee's training scripts:
   ```bash
   arcee-train --config spectrum-25.yml
   ```

## Resources

- [Arcee AI Documentation](https://arcee.ai/docs)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685) 