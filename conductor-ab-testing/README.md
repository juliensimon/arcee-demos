# LLM A/B Testing Tool: Compare AI Model Responses with Similarity Metrics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-5.23.3-orange)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)

A side-by-side comparison tool for evaluating AI model responses through [Arcee Conductor](https://conductor.arcee.ai) and [Together.ai](https://together.ai) APIs. Compare outputs, measure semantic similarity, collect human feedback, and make data-driven model selection decisions.

## Features

- **Side-by-Side Comparison** — Compare responses from any two models available through Arcee or Together.ai
- **Configurable Generation** — Control temperature, top_p, max_tokens, frequency_penalty, and presence_penalty
- **Semantic Similarity Metrics** — Quantitative analysis of response similarity using sentence transformers
- **Human Feedback Collection** — Save preferences for systematic model evaluation
- **Random Test Prompts** — Quickly evaluate models with pre-built prompts

## Available Models

### Arcee API
Auto-discovered from the Arcee API: virtuoso-large, virtuoso-medium, coder, spotlight, caller-large, blitz, and more.

### Together.ai
- **arcee-ai/AFM-4.5B** — Optimized 4.5B parameter model (requires `TOGETHER_API_KEY`)

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   CONDUCTOR_API_KEY=your_conductor_api_key_here
   TOGETHER_API_KEY=your_together_api_key_here
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Generation Parameters

Both models use identical parameters for fair comparison:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Temperature | 0.0–2.0 | 0.7 | Controls randomness |
| Top P | 0.0–1.0 | 0.9 | Nucleus sampling diversity |
| Max Tokens | 1–4000 | 1000 | Maximum response length |
| Frequency Penalty | -2.0–2.0 | 0.0 | Reduces token repetition |
| Presence Penalty | -2.0–2.0 | 0.0 | Encourages topic diversity |

## Deploy as a Hugging Face Space

1. Fork this repository
2. Create a new [Hugging Face Space](https://huggingface.co/spaces) with Gradio SDK
3. Add `CONDUCTOR_API_KEY` and `TOGETHER_API_KEY` as repository secrets
4. Push and deploy

## Adding Test Prompts

Create `test_prompts.json` in the project root:

```json
[
  "Explain quantum computing in simple terms",
  "Write a short poem about artificial intelligence",
  "What are the ethical implications of large language models?"
]
```

## Author

Built by [Julien Simon](https://julien.org). More on evaluating and comparing LLMs on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [Arcee Conductor](https://conductor.arcee.ai)
- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview)
- [Sentence Transformers](https://www.sbert.net/)

## License

MIT License — see the LICENSE file for details.
