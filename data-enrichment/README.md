# AI-Powered Medical Equipment Data Enrichment with Arcee AFM-4.5B

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![JSONL](https://img.shields.io/badge/Format-JSONL-lightgrey)](https://jsonlines.org/)
[![Arcee AI](https://img.shields.io/badge/AI-Arcee%20AFM--4.5B-green.svg)](https://arcee.ai/)

Enrich medical equipment metadata at scale using the [Arcee AFM-4.5B](https://arcee.ai) model. Transform basic product listings into comprehensive catalog entries with AI-generated descriptions, clinical applications, and risk assessments.

> *Note*: The original conductor demo is available in the `conductor-demo` branch.

## What It Does

Takes terse product names like `MASK SURG 3PLY ELASTIC EAR LOOP PLEATED DISP BFE98% ASTM2` and generates:

- **Descriptions** — Human-readable explanations of each product
- **Applications** — Primary clinical use cases and procedures
- **Risk Assessments** — Safety considerations and precautions

## Project Structure

```
data-enrichment/
├── metadata-enrichment.ipynb          # Main enrichment pipeline
├── metadata-enrichment-test-data.jsonl # Sample data (100 medical items)
├── enriched_lines.jsonl               # AI-enriched output
├── requirements.txt                   # Python dependencies
└── README.md
```

## Getting Started

1. Set up your environment:
   ```bash
   cd data-enrichment
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

2. Set your API key:
   ```bash
   export TOGETHER_API_KEY="your_api_key_here"
   ```

3. Launch Jupyter and open `metadata-enrichment.ipynb`:
   ```bash
   jupyter notebook
   ```

## Data Format

**Input (JSONL):**
```json
{
  "Item": "MASK SURG 3PLY ELASTIC EAR LOOP PLEATED DISP BFE98% ASTM2",
  "SKU": "8432710557",
  "Stock": 72,
  "LastUpdate": "2024-06-23T09:34:12Z"
}
```

**Output (JSONL):**
```json
{
  "Item": "MASK SURG 3PLY ELASTIC EAR LOOP PLEATED DISP BFE98% ASTM2",
  "SKU": "8432710557",
  "Stock": 72,
  "LastUpdate": "2024-06-23T09:34:12Z",
  "Description": "A 3-ply surgical mask with elastic ear loops...",
  "Applications": ["Hospital surgical procedures", "Protective wear in healthcare settings"],
  "Risks": ["Risk of infection if mask is not properly fitted"]
}
```

## Use Cases

- **Healthcare inventory management** — Enhanced search and discovery
- **Medical e-commerce** — Rich product listings
- **Clinical procurement** — Detailed specifications for purchasing
- **Compliance documentation** — Automated risk assessment generation

## Author

Built by [Julien Simon](https://julien.org). More on AI-powered data enrichment on the [AI Realist](https://www.airealist.ai) Substack.

## Resources

- [Arcee AI](https://arcee.ai)
- [Together.ai](https://together.ai/)
- [JSONL Format](https://jsonlines.org/)

---

**Note**: This tool is for demonstration purposes. Always verify AI-generated medical information with healthcare professionals for clinical applications.
