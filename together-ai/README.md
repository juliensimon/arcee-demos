# Arcee AI Models Demo with Together.ai

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Together.ai](https://img.shields.io/badge/Together-AI-green)](https://together.ai)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)

This repository contains demonstration notebooks and tools for using Arcee AI models through the Together.ai API platform, including a comprehensive domain-specific model evaluation tool.

## Prerequisites

- Python 3.8+
- A Together.ai API key (set as environment variable `TOGETHER_API_KEY`)

## Installation

```bash
pip install -r requirements.txt
```

## Radar Evaluator Tool

The `radar_evaluator.py` is a standalone tool for comparing language model performance across different industry domains. It features:

- **Zero notebook dependencies** - Pure Python script
- **Config-driven** - All model definitions and settings in `config.json`
- **Timestamped results** - Each evaluation creates a dedicated timestamped folder
- **Comprehensive output** - JSON, CSV, radar charts, and markdown reports
- **Parallel processing** - Multi-threaded evaluation for faster results

### Quick Start

```bash
# Compare AFM vs Llama3 on all industries
python radar_evaluator.py --model1 afm --model2 llama3_8b

# Compare specific models on specific industries with limited questions
python radar_evaluator.py --model1 gemma --model2 qwen --industries "Information Technology" "Healthcare" --num-questions 5
```

### Configuration

Edit `config.json` to:
- Add new models
- Modify evaluation parameters
- Adjust output settings
- Customize metrics

### Output Structure

Each evaluation creates a timestamped folder (`radar_results/YYYYMMDD_HHMMSS/`) containing:
- `model_comparison_results.json` - Raw evaluation data
- `radar_chart.png` - Visual comparison across industries
- `summary_report.md` - Statistical summary
- `detailed_results.csv` - Tabular data for analysis

### Programmatic Usage

```python
from radar_evaluator import RadarEvaluator

evaluator = RadarEvaluator("config.json")
results = evaluator.run_evaluation(
    model_1_key="afm",
    model_2_key="llama3_8b",
    industries=["Information Technology", "Healthcare"],
    num_questions=10
)
```

## Industry Questions

The `industry_questions.json` file contains 20 deep domain knowledge questions for each of the 10 major S&P 500 industries:

- Information Technology
- Healthcare
- Financials
- Consumer Discretionary
- Communication Services
- Industrials
- Consumer Staples
- Energy
- Utilities
- Materials

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

### 3. AFM Domain Evaluation (`afm_domains_demo.ipynb`)

Notebook for domain-specific model evaluation.

## Usage

1. Set your Together.ai API key as an environment variable:
   ```bash
   export TOGETHER_API_KEY='your_api_key_here'
   ```

2. For the radar evaluator:
   ```bash
   python radar_evaluator.py --help
   ```

3. For notebooks, open and run them in a Jupyter environment.

## Resources

- [Together.ai Documentation](https://docs.together.ai/)
- [Arcee AI Models](https://api.together.ai/models?q=arcee-ai)
- [Arcee AI Documentation](https://arcee.ai/docs) 