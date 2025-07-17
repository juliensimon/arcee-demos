# Arcee Model Engine Examples

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Arcee Model Engine](https://img.shields.io/badge/Arcee-Model%20Engine-purple)](https://arcee.ai)

Test notebooks for Arcee's model engine capabilities.

## Overview

This directory contains Jupyter notebooks and examples demonstrating Arcee's Model Engine capabilities. These notebooks showcase various features and use cases for Arcee's model engine, including auto-reasoning, tool use, and specialized model variants.

## Notebooks

- `test-model-engine-auto.ipynb`: Basic tests for Arcee's auto model capabilities
- `test-model-engine-auto-reasoning.ipynb`: Examples of auto-reasoning capabilities
- `test-model-engine-auto-tool.ipynb`: Demonstrations of tool usage with auto models
- `test-model-engine-caller.ipynb`: Tests for the Caller model variant
- `test-model-engine-coder.ipynb`: Examples using the Coder model variant
- `test-model-engine-spotlight.ipynb`: Tests for the Spotlight model variant
- `test-model-engine-virtuoso.ipynb`: Examples using the Virtuoso model variant

## Additional Files

- `print_streaming_response.py`: Utility for handling streaming responses
- `alice.txt` and `gatsby.txt`: Sample text files for testing
- `july14th.jpg`: Sample image for multimodal testing

## Requirements

To run these notebooks, you'll need:

1. Python 3.8+
2. Jupyter Notebook or JupyterLab
3. Arcee API credentials

## Getting Started

1. Set up your Python environment:
   ```bash
   python -m venv env-openai-client
   source env-openai-client/bin/activate  # On Windows: env-openai-client\Scripts\activate
   pip install jupyter openai pandas numpy matplotlib
   ```

2. Configure your Arcee API credentials as directed in the notebooks.

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

4. Open and run the notebooks.

## Model Variants

- **Auto**: General-purpose model with automatic capabilities
- **Caller**: Specialized for function calling and tool use
- **Coder**: Optimized for code generation and understanding
- **Spotlight**: Enhanced for specific focused tasks
- **Virtuoso**: High-performance general model with advanced capabilities

## Resources

- [Arcee AI Documentation](https://arcee.ai/docs)
- [Model Engine API Reference](https://arcee.ai/docs/model-engine) 