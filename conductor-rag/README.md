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

# Conductor RAG - Document Question-Answering System
🚀 A Retrieval-Augmented Generation (RAG) powered chat interface for document Q&A using Arcee Conductor

## Overview
This application provides an interactive chat interface that allows users to ask questions about their documents. It combines the power of Large Language Models with document retrieval to provide accurate, source-backed answers.

## Features
- **RAG-Powered Responses**: Leverages document context to provide accurate, factual answers
- **Flexible Query Modes**: Switch between RAG and vanilla LLM responses
- **Source Citations**: Automatically includes relevant document sources and page numbers
- **Interactive Interface**: Clean, user-friendly Gradio-based chat interface
- **Context Visibility**: View the retrieved document chunks used to generate responses

## Technical Details
- Built with Langchain and Gradio
- Uses Arcee Conductor API for LLM capabilities
- Document embedding via BAAI/bge-small-en-v1.5
- ChromaDB for vector storage
- Supports PDF document processing

## Included Papers
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

5. **Add Your Environment Variables**:
   The application uses the following environment variables, which you need to set in the Space settings:
   - `OPENAI_API_KEY`: Your Arcee Conductor API key

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

---
Built with 💖 using Arcee Conductor
