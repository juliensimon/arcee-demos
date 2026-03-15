# Run Trinity-Mini on Apple Silicon with MLX + OpenCode

A step-by-step guide to running Arcee AI's **Trinity-Mini** language model locally on your Mac using **MLX**, and connecting it to **OpenCode** as a fully local AI coding assistant.

**No API keys. No cloud. No per-token cost. Just your Mac.**

---

## When Local LLMs Are the Only Option

Some environments don't allow cloud AI — full stop. Air-gapped customer sites, classified networks, regulated industries (healthcare, finance, defense), or any situation where data sovereignty is non-negotiable. No internet means no API calls to Claude, GPT, or any hosted model.

A MacBook with Trinity-Mini runs entirely offline after the initial model download. The typical use cases are tasks where the data itself is the constraint — not the task complexity:

- Generating SQL queries against proprietary schemas you can't share with a cloud API
- Writing ETL scripts that reference confidential customer data structures
- Building validation pipelines over sensitive datasets
- Explaining and documenting legacy codebases during on-site audits
- Drafting IaC and deployment configs that embed internal network topology

---

## What You'll Build

```
┌─────────────┐     OpenAI-compatible API      ┌────────────────┐
│   OpenCode  │  ◄──────────────────────────►  │  mlx_lm.server │
│  (terminal  │    http://localhost:8080/v1    │  (Trinity-Mini │
│   coding AI)│                                │   on Metal GPU)│
└─────────────┘                                └────────────────┘
```

OpenCode talks to Trinity-Mini through a local OpenAI-compatible API. Everything runs on-device.

---

## Why This Stack?

| Component | What it does | Why it matters |
|-----------|-------------|----------------|
| **Trinity-Mini** | 26B-param MoE model (3B active) by Arcee AI, Apache 2.0 | [BFCL V3: 59.7](https://huggingface.co/arcee-ai/Trinity-Mini) (tool use — strong for its active parameter count), MMLU: 85.0, Math-500: 92.1 |
| **MLX** | Apple's ML framework for Apple Silicon | Native Metal GPU acceleration, no CUDA needed |
| **mlx_lm.server** | Serves MLX models via OpenAI-compatible API | Drop-in replacement for any OpenAI endpoint |
| **OpenCode** | Open-source terminal coding assistant (MIT) | Works with 75+ LLM providers including local endpoints; reads, edits, and runs code |

---

## How MLX and MoE Work Together

Apple Silicon uses **unified memory** — CPU and GPU share the same RAM. Unlike PyTorch/CUDA, where tensors must be copied across the PCIe bus to a discrete GPU, MLX accesses model weights in-place. For MoE models, this matters: all 128 experts sit in shared memory and the GPU can route to any of them without data movement.

```
Input token
     │
     ▼
┌─────────┐
│  Router │ ─── selects 8 of 128 experts
└─────────┘
     │
     ▼
┌─────────────────────┐
│ 8 active experts    │  ← Only these compute
│ (3B params total)   │
├─────────────────────┤
│ 120 idle experts    │  ← In memory but free
│ (23B params)        │
└─────────────────────┘
     │
     ▼
  Output token
```

You get the **knowledge** of a 26B model with the **speed** of a 3B model. The trade-off is memory: you need space for all 26B parameters even though most are idle.

---

## Choosing the Right Quantization

| Quantization | Peak Memory | Quality |
|-------------|------------|---------|
| **4-bit** | ~16 GB | Good |
| **5-bit** | ~19 GB | Good–Very good |
| **6-bit** | ~22 GB | Very good |
| **8-bit** | ~28 GB | Best |

**Recommendation:**
- **16 GB Mac:** 4-bit may work but expect memory pressure and swapping — 32 GB+ strongly recommended
- **32 GB Mac:** Use 4-bit, 5-bit, or 6-bit
- **48+ GB Mac:** Use 8-bit (no compromises)

These numbers may seem high for a "3B active" model — see [How MLX and MoE Work Together](#how-mlx-and-moe-work-together) for why all 26B parameters must be resident.

### Available MLX Models

All pre-converted by the [mlx-community](https://huggingface.co/mlx-community) on Hugging Face:

| Model | Link |
|-------|------|
| Trinity-Mini 8-bit | [mlx-community/Trinity-Mini-8bit](https://hf.co/mlx-community/Trinity-Mini-8bit) |
| Trinity-Mini 6-bit | [mlx-community/Trinity-Mini-6bit](https://hf.co/mlx-community/Trinity-Mini-6bit) |
| Trinity-Mini 5-bit | [mlx-community/Trinity-Mini-5bit](https://hf.co/mlx-community/Trinity-Mini-5bit) |
| Trinity-Mini 4-bit | [mlx-community/Trinity-Mini-4bit](https://hf.co/mlx-community/Trinity-Mini-4bit) |

> **What about Trinity Nano?** MLX versions of Trinity Nano (6B/1B active) exist on Hugging Face ([8-bit](https://hf.co/mlx-community/Trinity-Nano-Preview-8bit), [4-bit](https://hf.co/mlx-community/Trinity-Nano-Preview-4bit)) and work for simple chat. However, Nano's 1B active parameters are too small for reliable tool use — it struggles with OpenCode's structured tool schemas (missing parameters, malformed JSON). Use **Trinity Mini** for coding agents.

---

## Prerequisites

- **Mac with Apple Silicon** (M1/M2/M3/M4, any variant)
- **RAM:** See [Choosing the Right Quantization](#choosing-the-right-quantization) above
- **[uv](https://docs.astral.sh/uv/)** (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Homebrew** (recommended)

---

## Step 1: Enable Native Tool Calling

Before starting the server, apply the tool-calling patch. Out of the box, `mlx_lm.server` doesn't detect that Trinity models support tool calling. The model generates correct `<tool_call>` XML tags, but the server's streaming parser expects these to be single tokens in the vocabulary — and in the MLX-converted models, they're tokenized as multiple sub-tokens (`<`, `tool`, `_`, `call`, `>`).

This repo includes a patch script that fixes several issues:

1. **Removes the vocab guard** in `tokenizer_utils.py` that disables tool parsing when delimiters aren't single tokens
2. **Switches to buffer-based matching** in `server.py` so multi-token delimiters are detected as they accumulate
3. **Disables request batching** to work around a KV cache crash on concurrent requests ([#983](https://github.com/ml-explore/mlx-lm/issues/983))
4. **Adds `tool_parser_type`** to the model's `tokenizer_config.json`

```bash
./patch-tool-calling.sh          # defaults to 8-bit
./patch-tool-calling.sh 4bit     # or specify a quantization
```

The script ensures `mlx-lm` and the model are downloaded before applying patches. **First run downloads the model** from Hugging Face (~16 GB for 8-bit). Subsequent runs are instant.

Without the patch, OpenCode can still parse tool calls from the raw response, but the output includes visible XML tags and the API responses lack proper `finish_reason: tool_calls` signaling. It works, but the experience is rough. With the patch, you get clean structured tool calls.

> **Note:** The patch modifies files in uv's cache (`~/.cache/uv/`) and the Hugging Face model cache (`~/.cache/huggingface/`). It's safe to re-run and will be overwritten if you clear these caches. These fixes have been submitted upstream — see the [Appendix](#appendix-upstream-issues--pr) for details.

---

## Step 2: Start the Trinity-Mini Server

No install needed — `uvx` runs `mlx-lm` in an ephemeral environment:

```bash
uvx --from "mlx-lm>=0.28.4" mlx_lm.server \
    --model mlx-community/Trinity-Mini-8bit \
    --port 8080
```

Or use the launch script from this repo:

```bash
./start-server.sh 8bit    # or 4bit, 5bit, 6bit
```

If you ran Step 1 first, the model is already cached and the server starts immediately. Otherwise, the first run downloads the model (~16 GB for 8-bit).

> **Important:** Trinity-Mini uses the `afmoe` (Arcee Fused MoE) architecture, which requires **mlx-lm v0.28.4 or later**. The `uvx` command above handles this automatically.

> **What's happening:** `mlx_lm.server` loads the quantized MLX model into unified memory, then exposes it as an OpenAI-compatible REST API. The model weights are cached in `~/.cache/huggingface/hub/`. The server binds to `localhost` only — it's not accessible from other machines on your network. There is no authentication; if you need to expose it beyond localhost, put it behind a reverse proxy with auth.

---

## Step 3: Test the Server

In a **new terminal**:

```bash
./test-server.sh
```

Or manually with curl:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 100
  }'
```

You should get a JSON response with the model's completion. If you see it, the server is working.

---

## Step 4: Install OpenCode

```bash
# Recommended: Homebrew
brew install anomalyco/tap/opencode

# Alternative: npm
npm install -g opencode-ai

# Alternative: curl
curl -fsSL https://opencode.ai/install | bash
```

Verify:

```bash
opencode --version
```

---

## Step 5: Configure OpenCode for Trinity-Mini

This repo includes a ready-to-use `opencode.json`. Copy it to any project where you want to use Trinity-Mini:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "mlx/mlx-community/Trinity-Mini-8bit",
  "provider": {
    "mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX Local",
      "options": {
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        "mlx-community/Trinity-Mini-8bit": {
          "name": "Trinity Mini (MLX 8-bit)"
        },
        "mlx-community/Trinity-Mini-6bit": {
          "name": "Trinity Mini (MLX 6-bit)"
        },
        "mlx-community/Trinity-Mini-5bit": {
          "name": "Trinity Mini (MLX 5-bit)"
        },
        "mlx-community/Trinity-Mini-4bit": {
          "name": "Trinity Mini (MLX 4-bit)"
        }
      }
    }
  }
}
```

Or set it up globally (applies to all projects):

```bash
mkdir -p ~/.config/opencode
cp opencode.json ~/.config/opencode/opencode.json
```

> **How it works:** OpenCode uses the Vercel AI SDK's `@ai-sdk/openai-compatible` adapter to talk to `mlx_lm.server`. The `model` field format is `provider_id/model_id` — the model ID must match a key in the `models` map.

---

## Step 6: Launch OpenCode

Make sure the MLX server is running in another terminal (Step 2), then:

```bash
cd your-project
opencode
```

OpenCode will start in your terminal, connected to Trinity-Mini running locally on your Mac's GPU.

### Things to Try

```
> Write a SQL query to find duplicate customer records in the orders table
> Generate a Python script to validate CSV files against this JSON schema
> Explain how the auth middleware works in this codebase
> Write tests for the data migration module
```

---

## What to Expect

Trinity-Mini punches above its weight on tool use and structured output (see [benchmarks](#why-this-stack) — it beats models with similar active parameter counts). But with 3B active parameters, it's a significant step down from cloud models like Claude or GPT-4o. The model supports a **128K token context window**, though quality degrades on very long contexts — in practice OpenCode works best with a few files at a time rather than an entire large codebase. Expect:

- **Good results** on focused tasks: explaining code, writing simple functions, answering questions, summarizing text, tool calling
- **Mixed results** on complex tasks: multi-file refactoring, subtle bugs, long reasoning chains
- **More hallucinations** than cloud models — review generated code carefully

---

## Going Further

### Use Trinity-Mini with Other Tools

The `mlx_lm.server` API is OpenAI-compatible, so any tool that supports custom OpenAI endpoints can use the same local server:

- **Continue.dev** (VS Code): Point the OpenAI provider to `http://localhost:8080/v1`
- **Aider**: `aider --openai-api-base http://localhost:8080/v1`
- **curl/scripts**: Direct HTTP calls to the `/v1/chat/completions` endpoint

### Updating the Model

When `mlx-community` publishes a new quantization or Arcee releases a new Trinity version, update by changing the `--model` argument in `start-server.sh` (or the `uvx` command) and the model keys in `opencode.json`. The new weights download automatically on the next server start. To clean up old versions:

```bash
# List cached models
ls ~/.cache/huggingface/hub/models--mlx-community--Trinity-Mini-*

# Remove a specific version
rm -rf ~/.cache/huggingface/hub/models--mlx-community--Trinity-Mini-6bit
```

### Air-Gapped Setup

To run on a machine with no internet access, pre-download the model and runtime on a connected machine, then copy them over:

```bash
# On the connected machine: download everything
uvx --from "mlx-lm>=0.28.4" mlx_lm.server --model mlx-community/Trinity-Mini-8bit --port 8080
# Ctrl-C once it's loaded

# Transfer to the air-gapped machine
rsync -a ~/.cache/huggingface/hub/models--mlx-community--Trinity-Mini-8bit target:~/.cache/huggingface/hub/
rsync -a ~/.cache/uv/ target:~/.cache/uv/
```

### Compare with the Trinity Family

The full Trinity family runs on [Arcee AI's Trinity Arena](https://huggingface.co/spaces/arcee-ai/trinity-arena):
- **Trinity Nano** (6B/1B active) — fastest, lightest
- **Trinity Mini** (26B/3B active) — sweet spot for local use
- **Trinity Large** (400B/13B active) — maximum capability (cloud only)

---

## Appendix: Troubleshooting

### "Server not responding"

Make sure the server is running in another terminal. Check that nothing else is using port 8080:

```bash
lsof -i :8080
```

### "Model not found" in OpenCode

The model ID in `opencode.json` must match a key in the `models` map under the provider. If you see this error, check that the server is running and responding:

```bash
curl http://localhost:8080/v1/models
```

### Slow generation

- Close other memory-hungry apps (browsers, Docker)
- Try a smaller quantization (6-bit or 4-bit)
- Check memory pressure: `Activity Monitor > Memory > Memory Pressure`

### Out of memory

Switch to a smaller quantization:

```bash
./start-server.sh 4bit
```

### OpenCode hangs on first message

The first inference after starting the server can take 10-20 seconds as MLX compiles the Metal kernels. Subsequent messages will be faster.

---

## Appendix: Upstream Issues & PR

While building this walkthrough, we found two bugs in `mlx_lm.server` and reported them upstream to [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm):

| # | Issue | Description | Status |
|---|-------|-------------|--------|
| [#984](https://github.com/ml-explore/mlx-lm/issues/984) | Tool calling not detected for multi-token delimiters | `<tool_call>` tokenized as sub-tokens causes the server to disable tool parsing entirely | [PR #985](https://github.com/ml-explore/mlx-lm/pull/985) submitted |
| [#983](https://github.com/ml-explore/mlx-lm/issues/983) | KV cache crash on concurrent requests | `BatchRotatingKVCache.merge()` fails when batching requests with different prompt lengths | [PR #999](https://github.com/ml-explore/mlx-lm/pull/999) (fixes related #980) |

### PR #985: Fix tool call parsing for multi-token delimiters

**Changes:**
- **`tokenizer_utils.py`** — Vocab guard now warns instead of disabling tool calling, since the streaming parser can use text matching
- **`server.py`** — Streaming parser uses buffer-based substring matching instead of exact single-token matching

This affects all models where `<tool_call>` isn't a single special token in the vocabulary, including all `mlx-community/Trinity-*` conversions.

### Issue #983: KV cache crash on concurrent requests

When two requests with very different prompt lengths arrive simultaneously, `BatchRotatingKVCache.merge()` crashes with a shape mismatch (`ValueError: Shapes cannot be broadcast`). This affects models with sliding window attention (like Trinity). Workaround: the `patch-tool-calling.sh` script disables batching to prevent the crash.

[PR #999](https://github.com/ml-explore/mlx-lm/pull/999) fixes the related issue [#980](https://github.com/ml-explore/mlx-lm/issues/980) — prefix cache trimming failure for sliding window models. It separates physical vs logical offset tracking in `RotatingKVCache` and linearizes the ring buffer on trim, which addresses the underlying `BatchRotatingKVCache` instability for models like Trinity.

---

## Links

- [Trinity-Mini on Hugging Face](https://huggingface.co/arcee-ai/Trinity-Mini)
- [MLX Community Models](https://huggingface.co/mlx-community)
- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)
- [OpenCode](https://opencode.ai/)
- [Arcee AI](https://www.arcee.ai/)
