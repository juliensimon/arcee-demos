# Arcee Models on AWS SageMaker

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-orange)](https://aws.amazon.com/sagemaker/)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)

Deployment examples for Arcee models on AWS SageMaker.

## Overview

This directory contains Jupyter notebooks and examples for deploying Arcee AI models on AWS SageMaker. These examples demonstrate various deployment options, including CPU and GPU instances, with and without VLLM acceleration.

## Notebooks

- `deploy_agent_gpu.ipynb`: Deploy Arcee Agent on GPU instances
- `deploy_lite_cpu.ipynb`: Deploy Arcee Lite models on CPU instances
- `deploy_lite_gpu.ipynb`: Deploy Arcee Lite models on GPU instances
- `deploy_lite_gpu_vllm.ipynb`: Deploy Arcee Lite models on GPU with VLLM acceleration
- `deploy_llama_spark_gpu.ipynb`: Deploy Llama Spark models on GPU instances
- `deploy_nova_gpu.ipynb`: Deploy Nova models on GPU instances
- `deploy_scribe_gpu.ipynb`: Deploy Scribe models on GPU instances
- `deploy_spark_gpu.ipynb`: Deploy Spark models on GPU instances

## Additional Files

- `sagemaker_streaming.py`: Utility for handling streaming responses from SageMaker endpoints
- `machine-learning-wikipedia.txt`: Sample text for testing
- `chatgpt.png`: Sample image for multimodal testing

## Requirements

To run these notebooks, you'll need:

1. Python 3.8+
2. AWS account with SageMaker access
3. Appropriate IAM permissions
4. Jupyter Notebook or JupyterLab

## Getting Started

1. Set up your AWS credentials:
   ```bash
   aws configure
   ```

2. Launch SageMaker Studio or a SageMaker notebook instance.

3. Clone this repository and navigate to the `sagemaker` directory.

4. Open and run the notebooks.

## Deployment Options

- **CPU Deployment**: Cost-effective for smaller models or lower traffic
- **GPU Deployment**: Higher throughput and lower latency for production workloads
- **VLLM Acceleration**: Enhanced performance with VLLM (Variable Length Language Model) optimization
- **Model Variants**: Deploy different Arcee model variants based on your use case

## Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Arcee AI Documentation](https://arcee.ai/docs)
- [VLLM Documentation](https://vllm.readthedocs.io/) 