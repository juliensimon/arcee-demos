---
title: Conductor RAG
emoji: 🚀
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: "5.23.1"
app_file: app.py
pinned: false
---

# Local OpenAI Compatible Model + RAG - Document Question-Answering System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-5.23.1-orange)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)

🚀 A Retrieval-Augmented Generation (RAG) powered chat interface for document Q&A using a local OpenAI compatible model


*Note*: The original conductor demo is available in the conductor-demobranch.


## Overview
This application provides an interactive chat interface that allows users to ask questions about their documents and code. It combines the power of Large Language Models with document retrieval to provide accurate, source-backed answers from PDFs, text files, and code repositories.

## Features
- **RAG-Powered Responses**: Leverages document context to provide accurate, factual answers
- **Flexible Query Modes**: Switch between RAG and vanilla LLM responses
- **Source Citations**: Automatically includes relevant document sources and page numbers
- **Interactive Interface**: Clean, user-friendly Gradio-based chat interface
- **Context Visibility**: View the retrieved document chunks used to generate responses
- **Multi-Format Support**: Handles PDFs, text files, markdown, and code files
- **Code-Optimized Models**: Specialized embedding models for code understanding

## Technical Details
- Built with Langchain and Gradio
- Uses local OpenAI compatible model running on localhost:8080
- Document embedding via configurable models (including code-optimized options)
- ChromaDB for vector storage
- Supports PDF, text, markdown, and code file processing

## Setup and Usage

### 1. Setup Local Model

Before running the application, you need to have an OpenAI compatible model running locally on port 8080. Here are some popular options:

### Option 1: Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., Llama 3.1 8B)
ollama pull llama3.1:8b

# Run with OpenAI compatibility
ollama serve --host 0.0.0.0:8080
```

### Option 2: vLLM
```bash
# Install vLLM
pip install vllm

# Run OpenAI API server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8080
```

### Option 3: LM Studio
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load your preferred model
3. Start the local server on port 8080

### 2. Configuration (Optional)

You can customize the embedding model, text splitting, and paths by editing `config.json`:

```json
{
  "embeddings": {
    "model_name": "BAAI/bge-small-en-v1.5",
    "device": "cpu",
    "normalize_embeddings": true
  },
  "text_splitting": {
    "chunk_size": 512,
    "chunk_overlap": 128
  },
  "paths": {
    "vectorstore": "vectorstore",
    "pdf": "pdf"
  }
}
```

**Options:**
- `embeddings.model_name`: HuggingFace model name for embeddings
- `embeddings.device`: Device setting (ignored - auto-detection is used)
- `embeddings.normalize_embeddings`: Whether to normalize embeddings
- `text_splitting.chunk_size`: Size of text chunks (characters)
- `text_splitting.chunk_overlap`: Overlap between chunks (characters)
- `paths.vectorstore`: Directory for storing vector database
- `paths.pdf`: Directory containing PDF files

**Preset Models:**
The config includes preset models for easy selection:
- `general`: BAAI/bge-small-en-v1.5 (good for general text)
- `code_jina`: jinaai/jina-embeddings-v2-base-code (optimized for code)
- `code_salesforce_2b`: Salesforce/SFR-Embedding-Code-2B_R (large code model)
- `code_salesforce_400m`: Salesforce/SFR-Embedding-Code-400M_R (smaller code model)

**Quick Model Selection:**
To use a preset model, simply copy the model name from the presets:
```json
{
  "embeddings": {
    "model_name": "jinaai/jina-embeddings-v2-base-code"
  }
}
```

### 3. Data Ingestion

Before using the chatbot, you need to process your documents:

1. **Add your files** to the appropriate directories:
   - **PDF files**: `pdf/` directory
   - **Text/Code files**: `text/` directory (supports markdown, code, config files, etc.)
2. **Run the ingestion script**:
   ```bash
   python ingest.py
   ```
   This will:
   - Process all supported files in the configured directories
   - Create embeddings using your configured model
   - Store them in the vector database
   - Handle incremental updates for new documents
   - Show statistics about indexed files

3. **Analyze existing vector store** (optional):
   ```bash
   python ingest.py analyze
   ```
   This will show statistics about already indexed files without running ingestion.

**Supported file types:**
- **PDFs**: `.pdf`
- **Text files**: `.txt`, `.md`, `.markdown`
- **Code files**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, `.php`, `.rb`, `.swift`, `.kt`, `.scala`, `.sh`, `.sql`, `.html`, `.css`, `.json`, `.xml`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.conf`, `.log`
- **Hardware description**: `.vhd`, `.vhdl`, `.v`, `.sv`, `.svh`

### 4. Run the Chatbot

Once your documents are processed and your local model is running:

```bash
python app.py
```

This will start the Gradio web interface where you can chat with your documents and code.

## Included Documents
The following research papers are included in the `pdf` directory:

- [arXiv:2306.13649v3](https://arxiv.org/abs/2306.13649)
- [arXiv:2309.16609v1](https://arxiv.org/abs/2309.16609)
- [arXiv:2312.06795v1](https://arxiv.org/abs/2312.06795)
- [arXiv:2403.19522v1](https://arxiv.org/abs/2403.19522)
- [arXiv:2405.04434v5](https://arxiv.org/abs/2405.04434)
- [arXiv:2406.11617v1](https://arxiv.org/abs/2406.11617)
- [arXiv:2410.21228v1](https://arxiv.org/abs/2410.21228)
- [arXiv:2411.05059v2](https://arxiv.org/abs/2411.05059)
- [arXiv:2501.09223v1](https://arxiv.org/abs/2501.09223)
- [arXiv:2501.12948v1](https://arxiv.org/abs/2501.12948)
- [arXiv:2503.04872v1](https://arxiv.org/abs/2503.04872)

You can also add your own documents and code files to the respective directories.

## Deployment
This application is hosted as a Hugging Face Space. Configuration details can be found in the [spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference).

## Creating Your Own Hugging Face Space Using CLI

You can easily deploy this application as your own Hugging Face Space using the Hugging Face CLI. Follow these steps:

1. **Install the Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```

2. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   You'll be prompted to enter your Hugging Face token, which you can find in your account settings.

3. **Clone this Repository**:
   ```bash
   git clone https://github.com/username/conductor-rag.git
   cd conductor-rag
   ```

4. **Create a New Space**:
   ```bash
   huggingface-cli repo create conductor-rag-your-name --type space --space-sdk gradio
   ```

5. **Setup Local Model**:
   Ensure you have an OpenAI compatible model running locally on port 8080. You can use tools like:
   - [Ollama](https://ollama.ai/) with OpenAI compatibility
   - [vLLM](https://github.com/vllm-project/vllm) with OpenAI API server
   - [LM Studio](https://lmstudio.ai/) with OpenAI compatibility
   - Any other OpenAI compatible API server

7. **Push Your Code to the Space**:
   ```bash
   git remote add space https://huggingface.co/spaces/your-username/conductor-rag
   git push space main
   ```

8. **Add Your PDF Documents**:
   You can either add PDFs directly to the repository before pushing, or upload them later through git.

9. **Monitor Deployment**:
   Visit `https://huggingface.co/spaces/your-username/conductor-rag-your-name` to see your Space being built and deployed.

Your Space will automatically build and deploy the application. Once complete, you can access it via the provided URL and share it with others.

## Troubleshooting

### Apple Silicon (M1/M2) Macs
- **CPU usage**: The system automatically uses CPU on Apple Silicon Macs
- **Performance**: CPU performance is excellent on M1/M2 chips
- **Reliability**: Avoids GPU compatibility issues with embedding models

### Device Configuration
- **Automatic**: Device is automatically detected and selected
- **Priority order**: CUDA → CPU
- **No configuration needed**: Works out of the box on any system
- **Apple Silicon**: Uses CPU (reliable and fast for most models)

## Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces-overview)
- [Gradio Documentation](https://gradio.app/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Compatibility](https://platform.openai.com/docs/api-reference)

---
Built with 💖 using local AI models
