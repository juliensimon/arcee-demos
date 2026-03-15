# Build a Transparent Reasoning Chatbot with Arcee Trinity Mini

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-orange)](https://gradio.app/)
[![Arcee AI](https://img.shields.io/badge/Arcee-Trinity%20Mini-purple)](https://arcee.ai)

A dual-panel chatbot that makes AI reasoning visible in real time. Built with [Arcee AI](https://arcee.ai)'s Trinity Mini model (26B parameters, 3B active via Mixture of Experts), this demo streams the model's chain-of-thought reasoning alongside the final response — so you can watch the AI think step-by-step.

## How It Works

The interface shows two panels: one for the reasoning trace and one for the chat response. As Trinity Mini processes your prompt, you see both unfold in real time via streaming. This makes the model's decision-making transparent and debuggable.

## Contents

- `chatbot.py` — Gradio 6.x chatbot with dual-panel streaming (reasoning + response)
- `blog_post.md` — Write-up of the implementation
- `youtube_description.md` / `youtube_titles.md` — Video content
- `requirements.txt` — Python dependencies

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenRouter API key:
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```

3. Run the chatbot:
   ```bash
   python chatbot.py
   ```

## Key Features

- Real-time streaming of chain-of-thought reasoning
- Token limit controls for generation
- Conversation history
- Sample prompts for exploring model capabilities

## Author

Built by [Julien Simon](https://julien.org). More on transparent AI and reasoning models on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [Trinity Mini on Hugging Face](https://huggingface.co/arcee-ai/Trinity-Mini)
- [Arcee AI](https://arcee.ai)
- [OpenRouter](https://openrouter.ai/)
