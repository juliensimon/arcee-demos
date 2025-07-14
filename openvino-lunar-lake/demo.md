# OpenVINO LLM Inference Guide

This guide shows how to run Llama-3.1-SuperNova-Lite models using OpenVINO, from simple local inference to advanced server deployment.

## Table of Contents

1. [Model Export](#1-model-export-first-step)
2. [Quick Start](#2-quick-start-simplest)
3. [Local Inference](#3-local-inference-direct-model-access)
4. [OVMS Server Inference](#4-ovms-server-inference-multi-user)
5. [Direct API Calls](#5-direct-api-calls-advanced)
6. [Script Parameters](#6-script-parameters)
7. [Model Names for API Calls](#7-model-names-for-api-calls)
8. [Troubleshooting](#8-troubleshooting)
9. [Performance Comparison](#9-performance-comparison)
10. [Environment Setup](#10-environment-setup)

---

## 1. Model Export (First Step)

Only needed if you want to export different models or precisions.  
The models are already exported and ready to use.

### GPU Model Export (INT8 - Recommended)
```bash
python export_model.py text_generation --source_model arcee-ai/Llama-3.1-SuperNova-Lite --model_name supernova-gpu-int8 --target_device GPU --weight-format int8 --enable_prefix_caching --config_file_path models/config.json --ov_cache_dir ./ov_cache_dir --model_repository_path models --overwrite_model
```

### GPU Model Export (INT4 - Lightweight)
```bash
python export_model.py text_generation --source_model arcee-ai/Llama-3.1-SuperNova-Lite --model_name supernova-gpu-int4 --target_device GPU --weight-format int4 --enable_prefix_caching --config_file_path models/config.json --ov_cache_dir ./ov_cache_dir --model_repository_path models --overwrite_model
```

### CPU Model Export (INT8)
```bash
python export_model.py text_generation --source_model arcee-ai/Llama-3.1-SuperNova-Lite --model_name supernova-cpu-int8 --target_device CPU --weight-format int8 --enable_prefix_caching --config_file_path models/config.json --ov_cache_dir ./ov_cache_dir --model_repository_path models --overwrite_model
```

### CPU Model Export (INT4)
```bash
python export_model.py text_generation --source_model arcee-ai/Llama-3.1-SuperNova-Lite --model_name supernova-cpu-int4 --target_device CPU --weight-format int4 --enable_prefix_caching --config_file_path models/config.json --ov_cache_dir ./ov_cache_dir --model_repository_path models --overwrite_model
```

### NPU Model Export (INT4 - Channel-wise)
```bash
python export_model.py text_generation --source_model arcee-ai/Llama-3.1-SuperNova-Lite --model_name supernova-npu-int4-channel-wise --target_device NPU --weight-format int4 --enable_prefix_caching --config_file_path models/config.json --ov_cache_dir ./ov_cache_dir --model_repository_path models --overwrite_model
```

---

## 2. Quick Start (Simplest)

**Fastest way to test: Local inference on CPU**

```bash
python openvino_example.py --device CPU --precision int4 --prompt "Hello world"
```

---

## 3. Local Inference (Direct Model Access)

Local inference runs the model directly without a server.  
**Best for:** Single user, development, testing

### CPU Inference

**INT4 precision** (faster, smaller memory)
```bash
python openvino_example.py --device CPU --precision int4 --prompt "Explain quantum computing"
```

**INT8 precision** (better accuracy, larger memory)
```bash
python openvino_example.py --device CPU --precision int8 --prompt "What is machine learning?"
```

### GPU Inference

**INT4 precision** (recommended for GPU)
```bash
python openvino_example.py --device GPU --precision int4 --prompt "Generate a creative story"
```

**INT8 precision** (alternative for GPU)
```bash
python openvino_example.py --device GPU --precision int8 --prompt "Explain artificial intelligence"
```

### NPU Inference

**NPU uses optimized model** (precision parameter ignored)
```bash
python openvino_example.py --device NPU --prompt "What are the benefits of edge computing?"
```

---

## 4. OVMS Server Inference (Multi-User)

OVMS provides a server that multiple clients can connect to.  
**Best for:** Production, multiple users, API integration

### Option A: Auto-Start Server (Easiest)

The script starts the server, runs inference, then stops the server.

**CPU with INT4 - auto-start server**
```bash
python openvino_example.py --ovms --start-server --device CPU --precision int4 --prompt "Explain computer vision"
```

**GPU with INT4 - auto-start server**
```bash
python openvino_example.py --ovms --start-server --device GPU --precision int4 --prompt "Describe AI applications"
```

**NPU - auto-start server**
```bash
python openvino_example.py --ovms --start-server --device NPU --prompt "What are the benefits of edge computing?"
```

### Option B: Manual Server Management (Production)

Start server manually, run multiple inferences, stop when done.

**Step 1: Initialize OVMS environment (REQUIRED)**
```bash
.\ovms\setupvars.bat
```

**Step 2: Start server** (choose one based on your needs)
```bash
ovms/ovms --rest_port 9001 --config_path ./models/config-cpu-int4.json    # CPU INT4
ovms/ovms --rest_port 9001 --config_path ./models/config-cpu-int8.json    # CPU INT8
ovms/ovms --rest_port 9001 --config_path ./models/config-gpu-int4.json    # GPU INT4
ovms/ovms --rest_port 9001 --config_path ./models/config-gpu-int8.json    # GPU INT8
ovms/ovms --rest_port 9001 --config_path ./models/config-npu-int4.json    # NPU INT4
```

**Step 3: Run inference** (multiple times)
```bash
python openvino_example.py --ovms --device CPU --precision int4 --prompt "First prompt"
python openvino_example.py --ovms --device CPU --precision int4 --prompt "Second prompt"
```

**Step 4: Stop server when done**
```bash
Get-Process -Name ovms -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 5. Direct API Calls (Advanced)

Make direct HTTP requests to OVMS server.  
**Best for:** Custom applications, integration with other tools

### Prerequisites

1. OVMS server must be running (see section 3)
2. Use the correct model name based on your server config

### PowerShell Examples

**Basic completion request**
```powershell
$body = @{
    model = "supernova-gpu-int4"  # Use specific model name from your config
    max_tokens = 100
    temperature = 0.7
    stream = $false
    messages = @(
        @{role = "user"; content = "Explain quantum computing in simple terms"}
    )
} | ConvertTo-Json -Depth 3

Invoke-WebRequest -Uri "http://localhost:9001/v3/chat/completions" -Method POST -Body $body -ContentType "application/json" | Select-Object -ExpandProperty Content
```

**Streaming response**
```powershell
$body = @{
    model = "supernova-gpu-int4"
    max_tokens = 200
    temperature = 0.7
    stream = $true
    messages = @(
        @{role = "user"; content = "Write a short story about AI"}
    )
} | ConvertTo-Json -Depth 3

$response = Invoke-WebRequest -Uri "http://localhost:9001/v3/chat/completions" -Method POST -Body $body -ContentType "application/json"
$response.Content -split "`n" | ForEach-Object { if ($_ -match '^data: (.+)$') { $matches[1] } }
```

### CURL Examples

**PowerShell equivalent** (recommended for Windows)
```powershell
$curlBody = '{"model": "supernova-gpu-int4", "max_tokens": 50, "stream": false, "messages": [{"role": "user", "content": "Hello"}]}'
Invoke-WebRequest -Uri "http://localhost:9001/v3/chat/completions" -Method POST -Body $curlBody -ContentType "application/json"
```

---



## 6. Script Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--device` | CPU, GPU, NPU | GPU | Target device for inference |
| `--prompt` | Any text | "Deep learning is " | Input prompt for the model |
| `--precision` | int4, int8 | int4 | Model precision (ignored for NPU) |
| `--ovms` | Flag | False | Use OVMS server instead of local inference |
| `--ovms-port` | Port number | 9001 | OVMS server port |
| `--start-server` | Flag | False | Start OVMS server before inference |

---

## 7. Model Names for API Calls

When making direct API calls, use these exact model names:

### CPU Models
- `supernova-cpu-int8` (CPU INT8 - better accuracy)
- `supernova-cpu-int4` (CPU INT4 - lightweight)

### GPU Models
- `supernova-gpu-int8` (GPU INT8 - recommended)
- `supernova-gpu-int4` (GPU INT4 - lightweight)

### NPU Models
- `supernova-npu-int4-channel-wise` (NPU INT4 - optimized for edge)

---

## 8. Troubleshooting

### Common Issues

**Issue:** "Model with requested name is not found"  
**Solution:** Use the specific model name from section 7

**Issue:** "OVMS server is not accessible"  
**Solution:** 
1. Ensure OVMS server is running
2. Check the correct port (default: 9001)
3. Verify server started successfully

**Issue:** Server won't start  
**Solution:**
1. Run OVMS environment setup: `.\ovms\setupvars.bat`
2. Check if ports are already in use
3. Verify model paths exist
4. Check OVMS logs for specific errors

**Issue:** "HTTP/0.9 when not allowed"  
**Solution:** Use port 9001 (REST API) not 9000 (gRPC)

---

## 9. Performance Comparison

### Local Inference Performance Results

| Device | Precision | Words Generated | Time (seconds) | Speed (words/sec) | Performance Rank |
|--------|-----------|----------------|----------------|-------------------|------------------|
| **GPU** | INT4 | 403 | 38.60 | **10.4** | 🥇 Fastest |
| **NPU** | INT4 | 314 | 49.09 | 6.4 | 🥈 Second |
| **GPU** | INT8 | 351 | 67.73 | 5.2 | 🥉 Third |
| **CPU** | INT4 | 368 | 76.32 | 4.8 | 4th |
| **CPU** | INT8 | 380 | 90.70 | 4.2 | 5th |

### Key Performance Insights

- **GPU INT4 remains the fastest**: 10.4 words/second (2.5x faster than CPU INT8)
- **NPU shows strong performance**: 6.4 words/second, ranking second overall
- **INT4 generally outperforms INT8**: Lower precision provides better speed with acceptable quality
- **GPU vs CPU**: GPU provides significant speedup (2.2x faster for INT4)
- **NPU advantage**: Better performance than CPU options, optimized for edge computing

---

## 10. Environment Setup

**Activate virtual environment**
```bash
.\env\Scripts\activate
```

**IMPORTANT: Initialize OVMS environment** (REQUIRED for OVMS to work)
```bash
.\ovms\setupvars.bat
```

**Verify OpenVINO installation**
```bash
python -c "import openvino_genai; print('OpenVINO GenAI available')"
```

---



## Additional Resources

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Model export](https://docs.openvino.ai/2025/model-server/ovms_demos_common_export.html)
- [Local Inference Guide](https://docs.openvino.ai/2025/openvino-workflow-generative.html)
- [NPU Inference Guide](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) 