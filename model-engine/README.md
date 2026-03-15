# Arcee Model Engine: Auto, Coder, Virtuoso, and Caller Test Notebooks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Arcee Model Engine](https://img.shields.io/badge/Arcee-Model%20Engine-purple)](https://arcee.ai)

Jupyter notebooks for testing [Arcee AI](https://arcee.ai) Model Engine variants — Auto, Coder, Virtuoso, Caller, and Spotlight — covering auto-reasoning, tool use, code generation, and multimodal tasks.

## Notebooks

| Notebook | Model Variant | What It Demonstrates |
|----------|--------------|---------------------|
| `test-model-engine-auto.ipynb` | Auto | General-purpose model capabilities |
| `test-model-engine-auto-reasoning.ipynb` | Auto | Chain-of-thought reasoning |
| `test-model-engine-auto-tool.ipynb` | Auto | Tool use and function calling |
| `test-model-engine-caller.ipynb` | Caller | Specialized function calling |
| `test-model-engine-coder.ipynb` | Coder | Code generation and understanding |
| `test-model-engine-spotlight.ipynb` | Spotlight | Focused task completion |
| `test-model-engine-virtuoso.ipynb` | Virtuoso | High-performance general tasks |

## Additional Files

- `print_streaming_response.py` — Utility for handling streaming responses
- `alice.txt`, `gatsby.txt` — Sample texts for testing
- `july14th.jpg` — Sample image for multimodal testing

## Getting Started

1. Set up your Python environment:
   ```bash
   python -m venv env-openai-client
   source env-openai-client/bin/activate
   pip install jupyter openai pandas numpy matplotlib
   ```

2. Configure your Arcee API credentials as directed in the notebooks.

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## Author

Built by [Julien Simon](https://julien.org). More on evaluating language models on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [Arcee AI](https://arcee.ai)
- [Arcee Conductor](https://conductor.arcee.ai)
