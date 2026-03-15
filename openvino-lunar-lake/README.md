# Run Arcee AI Models on Intel Lunar Lake with OpenVINO

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2023.3+-green)](https://docs.openvino.ai/)
[![Intel](https://img.shields.io/badge/Intel-Lunar%20Lake-blue)](https://www.intel.com/)

Run optimized [Arcee AI](https://arcee.ai) models on Intel Lunar Lake processors using [OpenVINO](https://docs.openvino.ai/), leveraging the built-in NPU (Neural Processing Unit) for accelerated on-device inference.

## Contents

- `demo.md` — Step-by-step walkthrough
- `openvino_example.py` — Python script for running models with OpenVINO
- `requirements.txt` — Python dependencies

## Requirements

- Python 3.8+
- Intel Lunar Lake compatible hardware
- OpenVINO 2023.3+

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the example:
   ```bash
   python openvino_example.py
   ```

3. For a detailed walkthrough, see `demo.md`.

## Key Features

- Model optimization and quantization for Intel Lunar Lake
- NPU acceleration for efficient on-device inference
- Performance benchmarking across CPU, GPU, and NPU

## Author

Built by [Julien Simon](https://julien.org). More on running LLMs on edge hardware on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Intel Lunar Lake Processors](https://www.intel.com/content/www/us/en/products/docs/processors/lunar-lake-processors.html)
- [Arcee AI](https://arcee.ai)
