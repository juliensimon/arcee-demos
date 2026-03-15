# Benchmark Arcee AI Models on Together.ai: Domain-Specific LLM Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Together.ai](https://img.shields.io/badge/Together-AI-green)](https://together.ai)
[![Arcee AI](https://img.shields.io/badge/Arcee-AI-purple)](https://arcee.ai)

Demonstration notebooks and a standalone benchmarking tool for evaluating [Arcee AI](https://arcee.ai) models through the [Together.ai](https://together.ai) inference API — including domain-specific radar chart evaluations across S&P 500 industries.

## Prerequisites

- Python 3.8+
- A Together.ai API key (set as `TOGETHER_API_KEY`)

## Installation

```bash
pip install -r requirements.txt
```

## Radar Evaluator: Domain-Specific LLM Benchmarking Tool

The `radar_evaluator.py` is a standalone tool for comparing language model performance across industry domains. It produces radar charts, CSV reports, and JSON results.

### Quick Start

```bash
# Compare AFM vs Llama3 on all industries
python radar_evaluator.py --model1 afm --model2 llama3_8b

# Compare specific models on specific industries
python radar_evaluator.py --model1 gemma --model2 qwen --industries "Information Technology" "Healthcare" --num-questions 5
```

### Configuration

Edit `config.json` to add models, modify evaluation parameters, and customize metrics.

### Output

Each evaluation creates a timestamped folder (`radar_results/YYYYMMDD_HHMMSS/`) containing:
- `model_comparison_results.json` — Raw evaluation data
- `radar_chart.png` — Visual radar chart comparison
- `summary_report.md` — Statistical summary
- `detailed_results.csv` — Tabular data for analysis

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

## Industry Domains

The `industry_questions.json` file contains 20 domain-specific questions for each of the 10 major S&P 500 industries: Information Technology, Healthcare, Financials, Consumer Discretionary, Communication Services, Industrials, Consumer Staples, Energy, Utilities, and Materials.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `caller_demo.ipynb` | Function calling and tool use with the [Arcee Caller model](https://api.together.ai/models/arcee-ai/caller) |
| `virtuoso_demo.ipynb` | Text generation, streaming, and multi-turn chat with [Arcee Virtuoso Large](https://api.together.ai/models/arcee-ai/virtuoso-large) |
| `afm_domains_demo.ipynb` | Domain-specific evaluation of the AFM model |

## Getting Started

1. Set your API key:
   ```bash
   export TOGETHER_API_KEY='your_api_key_here'
   ```

2. Run the radar evaluator:
   ```bash
   python radar_evaluator.py --help
   ```

3. Or open the notebooks in Jupyter.

## Author

Built by [Julien Simon](https://julien.org). More on benchmarking and evaluating LLMs on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [Together.ai Documentation](https://docs.together.ai/)
- [Arcee AI Models on Together.ai](https://api.together.ai/models?q=arcee-ai)
- [Arcee AI](https://arcee.ai)
