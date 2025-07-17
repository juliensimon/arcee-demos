# Data Enrichment Tools

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![JSONL](https://img.shields.io/badge/Format-JSONL-lightgrey)](https://jsonlines.org/)

Tools for enriching metadata using AI.

## Overview

This directory contains tools and examples for enriching metadata using AI. The primary focus is on adding additional context, information, and structure to existing data using Arcee AI models.

## Contents

- `metadata-enrichment.ipynb`: Jupyter notebook demonstrating metadata enrichment techniques
- `metadata-enrichment-test-data.jsonl`: Sample test data in JSONL format
- `enriched_lines.jsonl`: Example output of enriched data

## Use Cases

- Adding semantic tags to content
- Extracting entities and relationships
- Generating summaries and descriptions
- Categorizing and classifying content
- Enhancing search and discovery capabilities

## Getting Started

1. Set up your Python environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install jupyter pandas numpy matplotlib
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `metadata-enrichment.ipynb` to explore the enrichment techniques.

## Data Format

The data is stored in JSONL (JSON Lines) format, where each line is a valid JSON object. This format is ideal for streaming processing and is commonly used for large datasets.

## Resources

- [JSONL Format Specification](https://jsonlines.org/)
- [Arcee AI Documentation](https://arcee.ai/docs) 