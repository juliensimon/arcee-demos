# Deploy Arcee AI Models on AWS SageMaker

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-orange)](https://aws.amazon.com/sagemaker/)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)

Step-by-step Jupyter notebooks for deploying [Arcee AI](https://arcee.ai) language models on [AWS SageMaker](https://aws.amazon.com/sagemaker/) — including CPU, GPU, and vLLM-accelerated configurations.

## Notebooks

| Notebook | Model | Instance Type |
|----------|-------|--------------|
| `deploy_agent_gpu.ipynb` | Arcee Agent | GPU |
| `deploy_lite_cpu.ipynb` | Arcee Lite | CPU |
| `deploy_lite_gpu.ipynb` | Arcee Lite | GPU |
| `deploy_lite_gpu_vllm.ipynb` | Arcee Lite | GPU + vLLM |
| `deploy_llama_spark_gpu.ipynb` | Llama Spark | GPU |
| `deploy_nova_gpu.ipynb` | Nova | GPU |
| `deploy_scribe_gpu.ipynb` | Scribe | GPU |
| `deploy_spark_gpu.ipynb` | Spark | GPU |

## Additional Files

- `sagemaker_streaming.py` — Utility for streaming responses from SageMaker endpoints
- `machine-learning-wikipedia.txt` — Sample text for testing
- `chatgpt.png` — Sample image for multimodal testing

## Requirements

- Python 3.8+
- AWS account with SageMaker access and appropriate IAM permissions
- Jupyter Notebook or JupyterLab

## Getting Started

1. Configure AWS credentials:
   ```bash
   aws configure
   ```

2. Launch SageMaker Studio or a notebook instance.

3. Clone and navigate to this directory, then open and run the notebooks.

## Deployment Options

- **CPU** — Cost-effective for smaller models or lower traffic
- **GPU** — Higher throughput and lower latency for production workloads
- **vLLM** — Optimized inference with PagedAttention for maximum throughput

## Author

Built by [Julien Simon](https://julien.org). More on deploying LLMs on AWS on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [Arcee AI](https://arcee.ai)
