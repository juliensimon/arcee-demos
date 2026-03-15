# Deploy Arcee AI Models on AWS Inferentia2

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-Inferentia2-orange)](https://aws.amazon.com/machine-learning/inferentia/)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)

Deploy and run [Arcee AI](https://arcee.ai) language models on [AWS Inferentia2](https://aws.amazon.com/machine-learning/inferentia/) custom AI accelerators for high-performance, cost-effective inference.

## Contents

- `llama-spark.py` — Deploy and compile Llama Spark models for Inferentia2
- `llama-spark-predict.py` — Run inference with compiled models on Inferentia2

## Requirements

- AWS account with access to Inferentia2 instances (e.g., `inf2.xlarge`)
- Python 3.8+
- AWS SDK for Python (Boto3)
- AWS Neuron SDK

## Getting Started

1. Set up the AWS Neuron SDK:
   ```bash
   . /etc/os-release
   sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOT
   deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
   EOT
   wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
   sudo apt-get update
   sudo apt-get install aws-neuronx-runtime-lib
   ```

2. Install Python dependencies:
   ```bash
   pip install torch-neuronx neuronx-cc
   ```

3. Run the example:
   ```bash
   python llama-spark.py
   ```

## Author

Built by [Julien Simon](https://julien.org). More on deploying AI models cost-effectively on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [AWS Inferentia2 Documentation](https://aws.amazon.com/machine-learning/inferentia/)
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)
- [Arcee AI](https://arcee.ai)
