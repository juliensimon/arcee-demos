# Deploy Arcee AFM-4.5B on Intel Xeon 6 (Granite Rapids) with OpenVINO Model Server

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-Model%20Server-green)](https://docs.openvino.ai/)
[![Intel](https://img.shields.io/badge/Intel-Xeon%206-blue)](https://www.intel.com/)
[![Arcee AI](https://img.shields.io/badge/Arcee-AFM--4.5B-purple)](https://arcee.ai)

Deploy [Arcee AI](https://arcee.ai)'s AFM-4.5B model on Intel Xeon 6 (Granite Rapids) processors using [OpenVINO Model Server](https://docs.openvino.ai/), with INT8 and INT4 quantization options for CPU-based inference.

## Contents

- `demo.txt` — Step-by-step deployment walkthrough
- `test_openai.py` — Test script using the OpenAI-compatible API
- `test_ov.py` — Test script using OpenVINO directly

## Prerequisites

- Intel Xeon 6 server (e.g., Amazon EC2 `r8i.4xlarge` or larger)
- Ubuntu 24
- Docker

## Quick Start

1. Set up the environment:
   ```bash
   virtualenv env && source env/bin/activate
   pip install optimum-intel[openvino]@git+https://github.com/huggingface/optimum-intel.git
   ```

2. Export the model with quantization:
   ```bash
   optimum-cli export openvino --model arcee-ai/AFM-4.5B --weight-format int8 afm_45b_ov_int8
   ```

3. Start the model server with Docker:
   ```bash
   docker run --rm -p 8000:8000 -v ~/models:/models:ro \
     openvino/model_server:latest \
     --port 9000 --rest_port 8000 --config_path /models/config_all.json
   ```

4. Query the model:
   ```bash
   curl -s http://localhost:8000/v3/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "afm_45b_ov_int8", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

For the full walkthrough, see `demo.txt`.

## Author

Built by [Julien Simon](https://julien.org). More on CPU-based LLM inference on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
- [Arcee AI](https://arcee.ai)
