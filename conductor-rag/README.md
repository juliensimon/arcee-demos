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

# RAG Document Q&A with Arcee Conductor, LangChain, and ChromaDB

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-5.23.1-orange)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![Arcee Conductor](https://img.shields.io/badge/Arcee-Conductor-purple)](https://conductor.arcee.ai)

A Retrieval-Augmented Generation (RAG) chat interface for document question-answering, built with [Arcee Conductor](https://conductor.arcee.ai), LangChain, ChromaDB, and Gradio. Upload PDFs and ask questions with source-backed, cited answers.

## Features

- **RAG-powered responses** — Answers grounded in your documents, not hallucinated
- **Source citations** — Every answer includes document sources and page numbers
- **Flexible query modes** — Switch between RAG and vanilla LLM responses
- **Context visibility** — View the retrieved document chunks used for each answer
- **Interactive chat** — Clean Gradio-based conversational interface

## Tech Stack

- [Arcee Conductor API](https://conductor.arcee.ai) for LLM capabilities
- [LangChain](https://python.langchain.com/) for RAG orchestration
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for document embeddings
- [Gradio](https://gradio.app/) for the web interface

## Included Papers

The `pdf` directory contains research papers for testing:

- [arXiv:2306.13649v3](https://arxiv.org/abs/2306.13649), [arXiv:2309.16609v1](https://arxiv.org/abs/2309.16609), [arXiv:2312.06795v1](https://arxiv.org/abs/2312.06795), [arXiv:2403.19522v1](https://arxiv.org/abs/2403.19522), [arXiv:2405.04434v5](https://arxiv.org/abs/2405.04434), [arXiv:2406.11617v1](https://arxiv.org/abs/2406.11617), [arXiv:2410.21228v1](https://arxiv.org/abs/2410.21228), [arXiv:2411.05059v2](https://arxiv.org/abs/2411.05059), [arXiv:2501.09223v1](https://arxiv.org/abs/2501.09223), [arXiv:2501.12948v1](https://arxiv.org/abs/2501.12948), [arXiv:2503.04872v1](https://arxiv.org/abs/2503.04872)

## Deploy as a Hugging Face Space

1. Install the Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. Create a new Space:
   ```bash
   huggingface-cli repo create conductor-rag --type space --space-sdk gradio
   ```

3. Add `OPENAI_API_KEY` (your Arcee Conductor API key) as a repository secret.

4. Push your code:
   ```bash
   git remote add space https://huggingface.co/spaces/your-username/conductor-rag
   git push space main
   ```

## Author

Built by [Julien Simon](https://julien.org). More on building RAG applications on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [Arcee Conductor](https://conductor.arcee.ai)
- [LangChain Documentation](https://python.langchain.com/)
- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview)
