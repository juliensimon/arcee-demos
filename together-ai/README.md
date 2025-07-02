# Arcee AI Models Demo with Together.ai

This repository contains demonstration notebooks for using Arcee AI models through the Together.ai API platform.

## Prerequisites

- Python 3.8+
- A Together.ai API key (set as environment variable `TOGETHER_API_KEY`)

## Installation

```bash
pip install -U pip
pip install -qU together yfinance
```

## Notebooks

### 1. Caller Model Demo (`caller_demo.ipynb`)

This notebook demonstrates how to use the [Arcee AI Caller model](https://api.together.ai/models/arcee-ai/caller) for function calling and tool use.

Features demonstrated:
- Function calling with structured tools
- Tool selection based on user queries
- Passing tool results to a larger model for comprehensive responses
- Error handling and fallbacks

### 2. Virtuoso Large Model Demo (`virtuoso_demo.ipynb`)

This notebook demonstrates how to use the [Arcee AI Virtuoso Large model](https://api.together.ai/models/arcee-ai/virtuoso-large) for various text generation tasks.

Features demonstrated:
- Basic text completion
- Chat completion
- Streaming responses
- Advanced prompting with system instructions
- Multi-turn conversation
- Direct API calls

## Usage

1. Set your Together.ai API key as an environment variable:
   ```python
   import os
   os.environ['TOGETHER_API_KEY'] = 'your_api_key_here'
   ```

2. Open and run the notebooks in a Jupyter environment.

3. Modify the example prompts and parameters to explore the models' capabilities.

## Resources

- [Together.ai Documentation](https://docs.together.ai/)
- [Arcee AI Models](https://api.together.ai/models?q=arcee-ai) 