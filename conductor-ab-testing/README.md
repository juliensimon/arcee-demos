---
title: Conductor A-B Testing
emoji: 🏆
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
---

# Conductor A-B Testing

![Conductor A-B Testing](https://img.shields.io/badge/Conductor-A--B%20Testing-green)

A powerful tool for comparing responses from different AI models through the Conductor API. This application allows you to conduct side-by-side comparisons, collect feedback, and analyze similarity metrics between model outputs.

## Features

- **Side-by-Side Model Comparison**: Compare responses from any two models available through the Conductor API
- **Detailed Metrics**: View token usage, response time, and multiple similarity metrics
- **Semantic Analysis**: Analyze the semantic similarity between model responses
- **Feedback Collection**: Save your preferences between model responses
- **Data Management**: View, copy, and reset collected feedback data
- **Random Prompts**: Load random test prompts to quickly evaluate models

## Getting Started

### Prerequisites

- Conductor API key - see https://conductor.arcee.ai
- Python 3.10+
- Required packages: gradio, openai, python-dotenv, sentence-transformers

### Installation

1. Clone this repository:
   ```bash
   git clone https://gitlab.com/juliensimon/arcee-demos.git
   cd conductor-ab-testing
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Conductor API key:
   ```
   CONDUCTOR_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Select two models from the dropdown menus
2. Enter your query or use the "Random Prompt" button
3. Click "Submit" to get responses from both models
4. View the responses and metrics
5. Provide feedback by clicking "Prefer this response" for your preferred model
6. View collected feedback data using the "Display Feedback Data" button

## Creating Your Own Space

You can easily deploy this application as a Hugging Face Space by following these steps:

1. Fork this repository to your GitHub account

2. Create a new Space on Hugging Face:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Connect your GitHub repository

3. Configure your Space:
   - Update the Space configuration in the README.md file:
     ```
     ---
     title: Conductor A-B Testing
     emoji: 🏆
     colorFrom: green
     colorTo: blue
     sdk: gradio
     sdk_version: 5.23.3
     app_file: app.py
     pinned: false
     ---
     ```

4. Add your Conductor API key as a secret:
   - Go to your Space settings
   - Navigate to the "Repository secrets" section
   - Add a new secret with the name `CONDUCTOR_API_KEY` and your API key as the value

5. Deploy your Space:
   - Commit and push your changes to GitHub
   - Hugging Face will automatically deploy your Space

6. Customize your Space:
   - Add a custom thumbnail image
   - Update the description
   - Add tags to make your Space discoverable

Your Space will be available at `https://huggingface.co/spaces/yourusername/your-space-name`

## Adding Test Prompts

Create a file named `test_prompts.json` in the project root with an array of prompts:

```json
[
  "Explain quantum computing in simple terms",
  "Write a short poem about artificial intelligence",
  "What are the ethical implications of large language models?",
  "Describe the process of photosynthesis"
]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Conductor API](https://conductor.arcee.ai) for providing access to various AI models
- [Gradio](https://gradio.app/) for the web interface framework
- [Sentence Transformers](https://www.sbert.net/) for semantic similarity analysis

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
