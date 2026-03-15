# Spectrum: LLM Quantization, LoRA, and Fine-Tuning Configurations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)

Configuration files and examples for model quantization and parameter-efficient fine-tuning using [Arcee AI](https://arcee.ai)'s Spectrum framework — including LoRA, QLoRA, FFT, and SVD-based compression.

## Contents

| File | Technique |
|------|-----------|
| `fft-8b.yaml` | 8-bit Fast Fourier Transform quantization |
| `lora-8b.yml` | 8-bit LoRA (Low-Rank Adaptation) |
| `qlora-8b.yml` | 8-bit QLoRA (Quantized LoRA) |
| `spectrum-25.yml` | Spectrum with 25% unfrozen parameters |
| `spectrum-50.yml` | Spectrum with 50% unfrozen parameters |
| `svd-example.py` | Singular Value Decomposition for model compression |
| `snr_results_*.json/yaml` | Signal-to-Noise Ratio analysis results |

## Techniques

- **Spectrum** — Selectively unfreeze model layers based on SNR analysis for efficient fine-tuning
- **LoRA / QLoRA** — Low-rank adaptation for parameter-efficient fine-tuning with optional quantization
- **FFT Quantization** — Frequency-domain quantization for model compression
- **SVD Compression** — Matrix factorization to reduce model size

## Getting Started

1. Install dependencies:
   ```bash
   pip install torch transformers peft
   ```

2. Run the SVD example:
   ```bash
   python svd-example.py
   ```

3. Use configs with Arcee's training scripts:
   ```bash
   arcee-train --config spectrum-25.yml
   ```

## Author

Built by [Julien Simon](https://julien.org). Deep dives on model optimization and quantization on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [Arcee AI](https://arcee.ai)
- [QLoRA Paper (arXiv:2305.14314)](https://arxiv.org/abs/2305.14314)
- [LoRA Paper (arXiv:2106.09685)](https://arxiv.org/abs/2106.09685)
