# Run Trinity Mini Locally on Apple Silicon with MLX — No Cloud, No API Keys

[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black)](https://support.apple.com/en-us/116943)
[![MLX](https://img.shields.io/badge/MLX-Framework-blue)](https://github.com/ml-explore/mlx)
[![Arcee AI](https://img.shields.io/badge/Arcee-Trinity%20Mini-purple)](https://arcee.ai)

Run [Arcee AI](https://arcee.ai)'s **Trinity Mini** (26B parameters, 3B active) entirely on your Mac using Apple's [MLX](https://github.com/ml-explore/mlx) framework. No API keys, no cloud, no per-token cost — ideal for air-gapped environments, regulated industries, and offline development.

This guide walks through setting up Trinity Mini as a local OpenAI-compatible API server, then connecting it to [OpenCode](https://opencode.ai/) as a fully local AI coding assistant.

## What You'll Build

```
┌─────────────┐     OpenAI-compatible API      ┌────────────────┐
│   OpenCode  │  ◄──────────────────────────►  │  mlx_lm.server │
│  (terminal  │    http://localhost:8080/v1    │  (Trinity-Mini │
│   coding AI)│                                │   on Metal GPU)│
└─────────────┘                                └────────────────┘
```

## Quantization Options

| Quantization | Peak Memory | Speed (M4 Max) | Quality |
|-------------|------------|-----------------|---------|
| **4-bit** | ~16 GB | 75.3 tok/s | Good |
| **5-bit** | ~19 GB | 67.9 tok/s | Good–Very good |
| **6-bit** | ~22 GB | 61.0 tok/s | Very good |
| **8-bit** | ~28 GB | 57.4 tok/s | Best |

**32 GB Mac:** Use 4-bit, 5-bit, or 6-bit. **48+ GB Mac:** Use 8-bit.

## Quick Start

1. Apply the tool-calling patch:
   ```bash
   ./patch-tool-calling.sh 8bit    # or 4bit, 5bit, 6bit
   ```

2. Start the server:
   ```bash
   ./start-server.sh 8bit
   ```

3. Test it:
   ```bash
   ./test-server.sh
   ```

4. Install and configure OpenCode:
   ```bash
   brew install anomalyco/tap/opencode
   cp opencode.json ~/.config/opencode/opencode.json
   opencode
   ```

## Contents

- `start-server.sh` — Launch the MLX server with any quantization
- `patch-tool-calling.sh` — Fix tool calling for multi-token delimiters
- `test-server.sh` — Verify the server is working
- `benchmark.sh` — Measure generation speed across quantizations
- `opencode.json` — Ready-to-use OpenCode configuration
- `schema.sql` — Sample SQL schema for testing
- `WALKTHROUGH.md` — Comprehensive step-by-step guide with technical details

For the full walkthrough including MLX/MoE internals, air-gapped setup, and troubleshooting, see [WALKTHROUGH.md](./WALKTHROUGH.md).

## Author

Built by [Julien Simon](https://julien.org). More on running LLMs locally on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [Trinity Mini on Hugging Face](https://huggingface.co/arcee-ai/Trinity-Mini)
- [MLX Community Models](https://huggingface.co/mlx-community)
- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)
- [OpenCode](https://opencode.ai/)
- [Arcee AI](https://arcee.ai)
