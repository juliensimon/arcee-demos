# Medical Equipment Data Enrichment with AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![JSONL](https://img.shields.io/badge/Format-JSONL-lightgrey)](https://jsonlines.org/)
[![Arcee AI](https://img.shields.io/badge/AI-Arcee%20AFM--4.5B-green.svg)](https://arcee.ai/)

AI-powered tools for enriching medical equipment metadata using the Arcee AFM-4.5B model. This project demonstrates how to enhance product catalogs with detailed descriptions, applications, and risk assessments.

## 🎯 Overview

This project showcases automated metadata enrichment for medical equipment catalogs. Using the Arcee AFM-4.5B model, it transforms basic product listings into comprehensive, AI-generated descriptions that include:

- **Detailed Descriptions**: Human-readable explanations of medical equipment
- **Applications**: Primary use cases and medical procedures
- **Risk Assessments**: Safety considerations and precautions

## 🚀 Features

- **AI-Powered Enrichment**: Uses Arcee AFM-4.5B for medical domain expertise
- **Batch Processing**: Handles large datasets efficiently with JSONL format
- **Medical Domain Focus**: Specialized for healthcare equipment and supplies
- **Structured Output**: Consistent JSON format for easy integration
- **Error Handling**: Robust processing with fallback mechanisms

## 📁 Project Structure

```
data-enrichment/
├── metadata-enrichment.ipynb          # Main Jupyter notebook with enrichment logic
├── metadata-enrichment-test-data.jsonl # Sample medical equipment data (100 items)
├── enriched_lines.jsonl               # AI-enriched output with descriptions
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Together AI API key (for Arcee model access)
- Jupyter Notebook

### Quick Start

1. **Clone and navigate to the project:**
   ```bash
   cd data-enrichment
   ```

2. **Set up your environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set your API key:**
   ```bash
   export TOGETHER_API_KEY="your_api_key_here"
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

5. **Open `metadata-enrichment.ipynb`** and run the cells to see the enrichment in action.

## 📊 Data Format

### Input Format (JSONL)
Each line contains a JSON object with basic medical equipment information:

```json
{
  "Item": "MASK SURG 3PLY ELASTIC EAR LOOP PLEATED DISP BFE98% ASTM2",
  "SKU": "8432710557",
  "Stock": 72,
  "LastUpdate": "2024-06-23T09:34:12Z"
}
```

### Output Format (JSONL)
Enriched data includes AI-generated fields:

```json
{
  "Item": "MASK SURG 3PLY ELASTIC EAR LOOP PLEATED DISP BFE98% ASTM2",
  "SKU": "8432710557",
  "Stock": 72,
  "LastUpdate": "2024-06-23T09:34:12Z",
  "Description": "A 3-ply surgical mask with elastic ear loops, pleated design, offering BFE98% filtration, and ASTM2 level of protection",
  "Applications": ["Hospital surgical procedures", "Laundry and material handling", "Protective wear in healthcare settings"],
  "Risks": ["Risk of infection if mask is not properly fitted and worn", "Potential for reduced efficacy if mask becomes wet or dirty", "Improper disposal may lead to environmental and health hazards"]
}
```

## 🔧 Usage

### Basic Enrichment

The notebook demonstrates the complete enrichment pipeline:

1. **Load Data**: Read JSONL file with medical equipment
2. **Process Items**: Send each item to Arcee AI for enrichment
3. **Handle Responses**: Parse AI-generated JSON responses
4. **Save Results**: Write enriched data to new JSONL file

### Customization

You can modify the enrichment by adjusting:

- **Model**: Change the Arcee model in the notebook
- **Prompt Engineering**: Modify the system and user prompts
- **Output Schema**: Adjust the expected JSON structure
- **Batch Size**: Process items in smaller/larger batches

## 🏥 Use Cases

### Healthcare Applications
- **Inventory Management**: Enhanced product descriptions for better search
- **Clinical Decision Support**: Detailed application and risk information
- **Training Materials**: Educational content for medical staff
- **Compliance**: Risk assessment documentation

### Business Applications
- **E-commerce**: Rich product listings for medical supply stores
- **Procurement**: Detailed specifications for purchasing decisions
- **Catalog Management**: Automated metadata generation
- **Search Optimization**: Improved discoverability through enhanced descriptions

## 🔍 Example Output

The enrichment process transforms basic product names into comprehensive descriptions:

**Before:**
```
"MASK SURG 3PLY ELASTIC EAR LOOP PLEATED DISP BFE98% ASTM2"
```

**After:**
```
Description: "A 3-ply surgical mask with elastic ear loops, pleated design, offering BFE98% filtration, and ASTM2 level of protection"
Applications: ["Hospital surgical procedures", "Laundry and material handling", "Protective wear in healthcare settings"]
Risks: ["Risk of infection if mask is not properly fitted and worn", "Potential for reduced efficacy if mask becomes wet or dirty", "Improper disposal may lead to environmental and health hazards"]
```

## ⚙️ Configuration

### Model Settings
- **Model**: `arcee-ai/AFM-4.5B`
- **Temperature**: Default (deterministic responses)
- **Max Tokens**: Auto-determined
- **Streaming**: Disabled for batch processing

### Error Handling
- JSON parsing fallbacks for malformed responses
- Retry logic for API failures
- Graceful degradation for processing errors

## 📈 Performance

- **Processing Speed**: ~2-3 seconds per item
- **Accuracy**: High-quality medical domain responses
- **Scalability**: Handles datasets of any size
- **Reliability**: Robust error handling and fallbacks

## 🔗 Resources

- [Arcee AI Documentation](https://arcee.ai/docs)
- [Together AI Platform](https://together.ai/)
- [JSONL Format Specification](https://jsonlines.org/)
- [Jupyter Notebook Documentation](https://jupyter.org/documentation)

## 🤝 Contributing

This project demonstrates AI-powered data enrichment techniques. Feel free to:

- Experiment with different prompts and models
- Extend the enrichment schema
- Add support for other data formats
- Improve error handling and performance

## 📄 License

This project is provided as a demonstration of AI-powered data enrichment capabilities. Please ensure compliance with your Together AI usage terms and any applicable data privacy regulations.

---

**Note**: This tool is designed for educational and demonstration purposes. Always verify AI-generated medical information and consult with healthcare professionals for clinical applications. 