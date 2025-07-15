#!/usr/bin/env python3
"""
Domain-Specific Model Evaluation Tool

A comprehensive tool for evaluating and comparing AI model performance across
different industry domains using automated scoring and visualization.
"""

__version__ = "1.0.0"
__author__ = "Arcee AI"

import argparse
import datetime
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from together import Together
from tqdm import tqdm

class RadarEvaluator:
    """
    A comprehensive evaluator for comparing AI model performance across industry domains.
    
    This class handles model evaluation, score extraction, visualization, and reporting
    for systematic comparison of AI model capabilities.
    """
    
    def __init__(self, config_path="config.json"):
        """Initialize the evaluator with configuration."""
        self.config = self.load_config(config_path)
        self.client = self.initialize_client()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(self.config["output"]["results_dir"]) / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading config from {config_path}: {e}")
            sys.exit(1)

    def initialize_client(self):
        """Initialize Together AI client."""
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            print("❌ Error: TOGETHER_API_KEY environment variable not set.")
            sys.exit(1)
        return Together(api_key=api_key)

    def load_industry_questions(self, path="industry_questions.json"):
        """Load industry questions from JSON file."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading industry questions from {path}: {e}")
            sys.exit(1)

    def get_system_prompt(self, model_name):
        """Generate system prompt for a model."""
        return f"""You are {model_name}, a helpful, harmless, and honest AI assistant developed by Arcee AI. 
You excel at providing accurate, comprehensive, and well-structured responses across various domains.
Your goal is to provide the most accurate, helpful, and thoughtful response possible to the user's query.
Always be respectful, educational, and informative in your responses."""

    def extract_scores_from_evaluation(self, evaluation_text):
        """Extract numerical scores from evaluation text."""
        scores = {metric: None for metric in self.config["metrics"].keys()}

        # Primary patterns - match each metric individually
        for metric in scores.keys():
            # Handle special case for "overall" metric which appears as "Overall Quality"
            if metric == "overall":
                pattern = r"Overall Quality:?\s*(\d+(?:\.\d+)?)/5"
            else:
                pattern = rf"{metric.capitalize()}:?\s*(\d+(?:\.\d+)?)/5"
            matches = re.search(pattern, evaluation_text, re.IGNORECASE)
            if matches:
                scores[metric] = float(matches.group(1))

        # Fallback patterns
        for metric in scores:
            if scores[metric] is None:
                # Handle special case for "overall" metric
                if metric == "overall":
                    pattern = r"Overall Quality[^\d]*(\d+(?:\.\d+)?)"
                else:
                    pattern = rf"{metric.capitalize()}[^\d]*(\d+(?:\.\d+)?)"
                matches = re.search(pattern, evaluation_text, re.IGNORECASE)
                if matches:
                    value = float(matches.group(1))
                    if (
                        self.config["metrics"][metric]["min"]
                        <= value
                        <= self.config["metrics"][metric]["max"]
                    ):
                        scores[metric] = value

        # Default values for missing scores and validation
        for metric in scores:
            if scores[metric] is None:
                print(f"No score found for {metric}")
                scores[metric] = 3.0
            elif not isinstance(scores[metric], (int, float)):
                print(f"Invalid score type for {metric}: {scores[metric]}")
                scores[metric] = 3.0
            elif (scores[metric] < self.config["metrics"][metric]["min"] or 
                  scores[metric] > self.config["metrics"][metric]["max"]):
                print(f"Score out of range for {metric}: {scores[metric]}")
                # Clamp to valid range
                scores[metric] = max(
                    self.config["metrics"][metric]["min"], 
                    min(self.config["metrics"][metric]["max"], scores[metric])
                )

        return scores

    def stream_model_response(self, model, messages, max_tokens=None, temperature=None):
        """Get streaming response from a model."""
        if max_tokens is None:
            max_tokens = self.config["evaluation"]["max_tokens"]
        if temperature is None:
            temperature = self.config["evaluation"]["temperature"]

        system_prompt = self.get_system_prompt(model["name"])
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        full_response = ""

        try:
            for chunk in self.client.chat.completions.create(
                model=model["model_endpoint"],
                messages=messages_with_system,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    if hasattr(chunk.choices[0], "delta"):
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            full_response += content

            if not full_response:
                response = self.client.chat.completions.create(
                    model=model["model_endpoint"],
                    messages=messages_with_system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                full_response = response.choices[0].message.content
        except Exception as e:
            full_response = f"Error: {str(e)}"

        return full_response

    def evaluate_with_deepseek(self, question, answer, model_name):
        """Evaluate a response using DeepSeek-R1."""
        evaluation_prompt = f"""Please evaluate the following answer to the question.
Rate the answer on these metrics (1-5 scale, where 5 is best):
1. Accuracy: How factually correct is the information?
2. Completeness: How thoroughly does it address all aspects of the question?
3. Clarity: How well-organized and easy to understand is the explanation?
4. Depth: How much relevant detail and insight does it provide?
5. Overall Quality: Your overall assessment of the response quality.

Question: {question}

Answer from {model_name}: {answer}

IMPORTANT: Please format your ratings in a consistent way using this exact format:
Accuracy: [score]/5
Completeness: [score]/5
Clarity: [score]/5
Depth: [score]/5
Overall Quality: [score]/5

Then provide your brief justification (2-3 sentences) for your assessment below the ratings."""

        try:
            deepseek_response = self.client.chat.completions.create(
                model=self.config["evaluation"]["evaluator_model"],
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=self.config["evaluation"]["evaluation_max_tokens"],
                temperature=self.config["evaluation"]["evaluation_temperature"],
            )
            evaluation = deepseek_response.choices[0].message.content
            scores = self.extract_scores_from_evaluation(evaluation)
        except Exception as e:
            evaluation = f"Error: {str(e)}"
            scores = {k: 3.0 for k in self.config["metrics"].keys()}

        return evaluation, scores
    
    def validate_scores(self, scores):
        """Validate that all required scores are present and within valid range."""
        required_metrics = set(self.config["metrics"].keys())
        present_metrics = set(scores.keys())
        
        if required_metrics != present_metrics:
            missing = required_metrics - present_metrics
            extra = present_metrics - required_metrics
            print(f"Warning: Score validation failed - missing: {missing}, extra: {extra}")
            return False
        
        for metric, score in scores.items():
            if not isinstance(score, (int, float)):
                print(f"Warning: Invalid score type for {metric}: {score}")
                return False
            if score < 1 or score > 5:
                print(f"Warning: Score out of range for {metric}: {score}")
                return False
        
        return True

    def process_question(self, model_1, model_2, question, industry, prompt_idx):
        """Process a single question with two models."""
        messages = [{"role": "user", "content": question}]

        # Get responses from both models
        result_1 = self.stream_model_response(model_1, messages)
        eval_1, scores_1 = self.evaluate_with_deepseek(
            question, result_1, model_1["name"]
        )
        
        # Validate scores
        if not self.validate_scores(scores_1):
            print(f"Warning: Invalid scores for {model_1['name']} on question {prompt_idx}")

        result_2 = self.stream_model_response(model_2, messages)
        eval_2, scores_2 = self.evaluate_with_deepseek(
            question, result_2, model_2["name"]
        )
        
        # Validate scores
        if not self.validate_scores(scores_2):
            print(f"Warning: Invalid scores for {model_2['name']} on question {prompt_idx}")

        return {
            "industry": industry,
            "prompt_idx": prompt_idx,
            "model_1": {"response": result_1, "evaluation": eval_1, "scores": scores_1},
            "model_2": {"response": result_2, "evaluation": eval_2, "scores": scores_2},
            "question": question,
        }

    def generate_radar_chart(self, results, model_1_name, model_2_name):
        """Generate radar chart comparing model performance."""
        if not results:
            # Create empty figure if no results
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No results available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, style='italic')
            return fig
            
        # Aggregate scores by industry
        industries = list(set([r["industry"] for r in results]))
        metrics = list(self.config["metrics"].keys())

        model_1_scores = {
            industry: {metric: [] for metric in metrics} for industry in industries
        }
        model_2_scores = {
            industry: {metric: [] for metric in metrics} for industry in industries
        }

        for result in results:
            industry = result["industry"]
            for metric in metrics:
                model_1_scores[industry][metric].append(
                    result["model_1"]["scores"][metric]
                )
                model_2_scores[industry][metric].append(
                    result["model_2"]["scores"][metric]
                )

        # Calculate averages with error handling for empty lists
        model_1_avg = {}
        model_2_avg = {}
        
        for industry in industries:
            model_1_avg[industry] = {}
            model_2_avg[industry] = {}
            for metric in metrics:
                scores_1 = model_1_scores[industry][metric]
                scores_2 = model_2_scores[industry][metric]
                model_1_avg[industry][metric] = np.mean(scores_1) if scores_1 else 0.0
                model_2_avg[industry][metric] = np.mean(scores_2) if scores_2 else 0.0

        # Create radar chart - dynamically size based on number of industries
        num_industries = len(industries)
        if num_industries <= 5:
            rows, cols = 1, num_industries
        elif num_industries <= 10:
            rows, cols = 2, 5
        else:
            # Ceiling division for larger numbers
            rows, cols = (num_industries + 4) // 5, 5
        
        fig, axes = plt.subplots(
            rows, cols, 
            figsize=(4*cols, 3*rows), 
            subplot_kw=dict(projection="polar")
        )
        if num_industries == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, industry in enumerate(industries):
            ax = axes[i]

            # Prepare data for radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            model_1_values = [model_1_avg[industry][metric] for metric in metrics]
            model_2_values = [model_2_avg[industry][metric] for metric in metrics]
            model_1_values += model_1_values[:1]
            model_2_values += model_2_values[:1]

            # Plot
            ax.plot(
                angles,
                model_1_values,
                "o-",
                linewidth=2,
                label=model_1_name,
                color="blue",
            )
            ax.fill(angles, model_1_values, alpha=0.25, color="blue")
            ax.plot(
                angles,
                model_2_values,
                "o-",
                linewidth=2,
                label=model_2_name,
                color="red",
            )
            ax.fill(angles, model_2_values, alpha=0.25, color="red")

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.capitalize() for m in metrics])
            ax.set_ylim(0, 5)
            ax.set_title(industry, size=12, y=1.08)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        # Hide unused subplots
        for i in range(num_industries, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig

    def generate_summary_report(self, results, model_1_name, model_2_name):
        """Generate a summary report of the evaluation."""
        if not results:
            return "# Model Comparison Report\n\nNo results available for comparison."
        
        # Calculate overall statistics
        model_1_overall_scores = []
        model_2_overall_scores = []

        for result in results:
            model_1_overall_scores.append(result["model_1"]["scores"]["overall"])
            model_2_overall_scores.append(result["model_2"]["scores"]["overall"])

        report = f"""
# Model Comparison Report
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Models Evaluated
- Model 1: {model_1_name}
- Model 2: {model_2_name}

## Overall Performance
- {model_1_name} Average Score: {np.mean(model_1_overall_scores):.2f} ± {np.std(model_1_overall_scores):.2f}
- {model_2_name} Average Score: {np.mean(model_2_overall_scores):.2f} ± {np.std(model_2_overall_scores):.2f}
- Total Questions Evaluated: {len(model_1_overall_scores)}

## Industry Breakdown
"""

        industries = list(set([r["industry"] for r in results]))
        for industry in industries:
            industry_results = [r for r in results if r["industry"] == industry]
            model_1_scores = [
                r["model_1"]["scores"]["overall"] for r in industry_results
            ]
            model_2_scores = [
                r["model_2"]["scores"]["overall"] for r in industry_results
            ]

            report += f"""
### {industry}
- {model_1_name}: {np.mean(model_1_scores):.2f} ± {np.std(model_1_scores):.2f} (n={len(model_1_scores)})
- {model_2_name}: {np.mean(model_2_scores):.2f} ± {np.std(model_2_scores):.2f} (n={len(model_2_scores)})
"""

        return report

    def save_results(self, results, model_1_name, model_2_name):
        """Save all results to the timestamped directory."""
        # Save raw results
        results_file = self.results_dir / "model_comparison_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Generate and save radar chart
        fig = self.generate_radar_chart(results, model_1_name, model_2_name)
        chart_file = self.results_dir / "radar_chart.png"
        fig.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Generate and save summary report
        report = self.generate_summary_report(results, model_1_name, model_2_name)
        report_file = self.results_dir / "summary_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        # Save detailed CSV
        csv_data = []
        for result in results:
            row = {
                "industry": result["industry"],
                "prompt_idx": result["prompt_idx"],
                "question": result["question"],
                f"{model_1_name}_response": result["model_1"]["response"],
                f"{model_1_name}_evaluation": result["model_1"]["evaluation"],
                f"{model_2_name}_response": result["model_2"]["response"],
                f"{model_2_name}_evaluation": result["model_2"]["evaluation"],
            }

            # Add scores
            for metric in self.config["metrics"].keys():
                row[f"{model_1_name}_{metric}"] = result["model_1"]["scores"][metric]
                row[f"{model_2_name}_{metric}"] = result["model_2"]["scores"][metric]

            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_file = self.results_dir / "detailed_results.csv"
        df.to_csv(csv_file, index=False)

        print(f"\n✅ Results saved to: {self.results_dir}")
        print(f"📊 Raw results: {results_file}")
        print(f"📈 Radar chart: {chart_file}")
        print(f"📋 Summary report: {report_file}")
        print(f"📄 Detailed CSV: {csv_file}")

    def run_evaluation(
        self, model_1_key, model_2_key, industries=None, num_questions=None
    ):
        """Run the complete evaluation."""
        model_1 = self.config["models"][model_1_key]
        model_2 = self.config["models"][model_2_key]

        industry_questions = self.load_industry_questions()
        selected_industries = (
            industries if industries else list(industry_questions.keys())
        )

        # Calculate total number of questions across all industries
        total_questions = 0
        for industry in selected_industries:
            questions = industry_questions[industry]
            if num_questions:
                questions = questions[:num_questions]
            total_questions += len(questions)
        
        print(f"\n🔧 Preparing evaluation for {len(selected_industries)} industries...")
        for industry in selected_industries:
            questions = industry_questions[industry]
            if num_questions:
                questions = questions[:num_questions]
            print(f"  📝 {industry}: {len(questions)} questions")

        print(f"\n🚀 Starting evaluation of {total_questions} total questions...")
        
        # Process industries with a single progress bar
        results = []
        with tqdm(total=total_questions, desc="Overall Progress") as pbar:
            for industry in selected_industries:
                questions = industry_questions[industry]
                if num_questions:
                    questions = questions[:num_questions]

                with ThreadPoolExecutor(
                    max_workers=self.config["output"]["max_workers"]
                ) as executor:
                    futures = [
                        executor.submit(
                            self.process_question, model_1, model_2, q, industry, i
                        )
                        for i, q in enumerate(questions)
                    ]

                    for f in as_completed(futures):
                        res = f.result()
                        results.append(res)
                        pbar.update(1)

        # Save results
        self.save_results(results, model_1["name"], model_2["name"])

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Domain-Specific Model Evaluation Tool",
        epilog="Example: python radar_evaluator.py --model1 afm --model2 gemma --num-questions 5"
    )
    parser.add_argument(
        "--model1",
        type=str,
        default="afm",
        help="First model key (available: afm, gemma, qwen, llama3_8b)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="llama3_8b",
        help="Second model key (available: afm, gemma, qwen, llama3_8b)",
    )
    parser.add_argument(
        "--industries",
        type=str,
        nargs="+",
        default=None,
        help="Industries to evaluate (default: all)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions per industry (default: all)",
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Configuration file path"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RadarEvaluator(args.config)

    # Validate model keys
    if args.model1 not in evaluator.config["models"]:
        print(f"❌ Error: Model '{args.model1}' not found in config")
        sys.exit(1)
    if args.model2 not in evaluator.config["models"]:
        print(f"❌ Error: Model '{args.model2}' not found in config")
        sys.exit(1)

    # Run evaluation
    results = evaluator.run_evaluation(
        args.model1, args.model2, args.industries, args.num_questions
    )

    print(f"\n🎉 Evaluation completed! Results saved to: {evaluator.results_dir}")


if __name__ == "__main__":
    main()
