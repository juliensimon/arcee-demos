# AWS Inferentia2 Examples

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-Inferentia2-orange)](https://aws.amazon.com/machine-learning/inferentia/)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)

Examples for running Arcee models on AWS Inferentia2 accelerators.

## Overview

This directory contains examples and scripts for deploying and running Arcee AI models on AWS Inferentia2 accelerators. Inferentia2 is AWS's custom-designed machine learning chip that provides high-performance, cost-effective inference for deep learning models.

## Contents

- `llama-spark.py`: Script for deploying and running Llama models with Spark on Inferentia2
- `llama-spark-predict.py`: Prediction script for Llama models on Inferentia2

## Requirements

- AWS account with access to Inferentia2 instances
- Python 3.8+
- AWS SDK for Python (Boto3)
- AWS Neuron SDK

## Getting Started

1. Set up the AWS Neuron SDK:
   ```bash
   # For Ubuntu 20.04
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

## Resources

- [AWS Inferentia Documentation](https://aws.amazon.com/machine-learning/inferentia/)
- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)
- [Arcee AI Documentation](https://arcee.ai/docs) 