# OpenVINO on Intel Lunar Lake

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2023.3+-green)](https://docs.openvino.ai/)
[![Intel](https://img.shields.io/badge/Intel-Lunar%20Lake-blue)](https://www.intel.com/)

Demonstrations for Intel's Lunar Lake with OpenVINO.

## Overview

This directory contains examples and demonstrations for running AI models on Intel's Lunar Lake processors using OpenVINO. Lunar Lake is Intel's next-generation client processor architecture with enhanced AI capabilities, and OpenVINO is Intel's toolkit for optimizing and deploying deep learning models.

## Contents

- `demo.md`: Detailed demonstration walkthrough
- `openvino_example.py`: Python example for running models with OpenVINO
- `requirements.txt`: Required Python dependencies

## Requirements

- Python 3.8+
- Intel Lunar Lake compatible hardware
- OpenVINO 2023.3+

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the example:
   ```bash
   python openvino_example.py
   ```

3. For a detailed walkthrough, refer to `demo.md`.

## Key Features

- Model optimization for Intel Lunar Lake
- Leveraging NPU (Neural Processing Unit) for acceleration
- Efficient inference with OpenVINO runtime
- Performance benchmarking and comparison

## Resources

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Intel Lunar Lake Information](https://www.intel.com/content/www/us/en/products/docs/processors/lunar-lake-processors.html)
- [Arcee AI Documentation](https://arcee.ai/docs) 