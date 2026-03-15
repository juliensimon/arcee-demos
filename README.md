# Arcee AI Demos: Deploy, Fine-Tune, and Evaluate Small Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)
[![Conductor](https://img.shields.io/badge/Conductor-API-green)](https://conductor.arcee.ai)

Hands-on examples for deploying, evaluating, and optimizing [Arcee AI](https://arcee.ai) language models — including the Trinity model family, AFM, Virtuoso, and more — across AWS, Intel, Apple Silicon, and cloud API platforms.

Whether you're looking to run LLMs locally on a MacBook, deploy on AWS SageMaker, benchmark model quantizations, or build RAG applications, this repository has a working demo for you.

## Repository Structure

| Demo | Description |
|------|-------------|
| [**trinity-mini**](./trinity-mini/) | Build a transparent reasoning chatbot with Arcee Trinity Mini on OpenRouter |
| [**trinity-mini-mlx**](./trinity-mini-mlx/) | Run Trinity Mini locally on Apple Silicon with MLX — no cloud, no API keys |
| [**conductor-rag**](./conductor-rag/) | Retrieval-Augmented Generation (RAG) document Q&A app with Arcee Conductor |
| [**conductor-ab-testing**](./conductor-ab-testing/) | A/B testing tool for comparing LLM responses with semantic similarity metrics |
| [**data-enrichment**](./data-enrichment/) | AI-powered medical equipment metadata enrichment with Arcee AFM-4.5B |
| [**together-ai**](./together-ai/) | Domain-specific LLM benchmarking with radar charts via Together.ai |
| [**sagemaker**](./sagemaker/) | Deploy Arcee models on AWS SageMaker (CPU, GPU, vLLM) |
| [**inferentia2**](./inferentia2/) | Run Arcee models on AWS Inferentia2 custom AI accelerators |
| [**openvino-lunar-lake**](./openvino-lunar-lake/) | Optimize and run models on Intel Lunar Lake with OpenVINO |
| [**openvino-granite-rapids**](./openvino-granite-rapids/) | Deploy AFM-4.5B on Intel Xeon 6 (Granite Rapids) with OpenVINO Model Server |
| [**spectrum**](./spectrum/) | Model quantization and fine-tuning configs (LoRA, QLoRA, Spectrum, SVD) |
| [**model-engine**](./model-engine/) | Test notebooks for Arcee Model Engine variants (Auto, Coder, Virtuoso, Caller) |
| [**arcee-agent**](./arcee-agent/) | AI agent demos with tool use (Yahoo Finance assistant) |
| [**anymcp**](./anymcp/) | Arcee AnyMCP integration examples |

## Getting Started

Each subdirectory contains its own README with setup instructions. Generally, you'll need:

1. Python 3.8+
2. Required API keys (Arcee Conductor, Together.ai, OpenRouter, etc.)
3. Dependencies installed via `pip install -r requirements.txt` in the respective directory

## Author

Built by [Julien Simon](https://julien.org). Read more about practical AI and small language models on the [AI Realist](https://www.airealist.ai) Substack.

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Resources

- [Arcee AI](https://arcee.ai) — Small language models for enterprise
- [Conductor API](https://conductor.arcee.ai) — Arcee's model serving platform
- [Together.ai](https://docs.together.ai/) — Cloud inference API
- [AI Realist](https://www.airealist.ai) — Julien's Substack on practical AI
- [julien.org](https://julien.org) — Author's website
