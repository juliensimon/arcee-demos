"""
A/B Testing Application for Conductor API Models

This application allows users to compare responses from different AI models
through the Conductor API, collect feedback, and analyze similarity metrics.
"""

# Standard library imports
import os
import os.path
import time
import json
import uuid
import logging
import random
from typing import List, Dict, Any


import openai
from similarity import (
    jaccard_similarity,
    cosine_similarity,
    levenshtein_similarity,
    semantic_similarity,
)
import concurrent.futures
import re


# Set up logging
logging.basicConfig(level=logging.INFO)

# Set the base URL for the Conductor API
CONDUCTOR_API_BASE = "https://conductor.arcee.ai/v1"

# Path to the feedback JSON file
FEEDBACK_FILE = "user_feedback.json"

# Import the model loader
from model_loader import get_model

# Current request ID and prompt (global to track current session)
current_request = {
    "id": None,
    "prompt": None,
    "model_a": None,
    "model_b": None,
    "response_a": None,
    "response_b": None,
}

# Cache for available models
cached_models = None


def get_available_models(force_refresh: bool = False) -> List[str]:
    """
    Query the Conductor API to get a list of available models.
    Uses a cache to avoid repeated API calls.

    Args:
        force_refresh (bool): If True, forces a refresh of the cached models

    Returns:
        List[str]: List of model names
    """
    global cached_models

    # Return cached models if available and not forcing refresh
    if cached_models is not None and not force_refresh:
        return cached_models

    try:
        # Get API key from environment variable
        api_key = os.environ.get("CONDUCTOR_API_KEY")
        if not api_key:
            logging.error("CONDUCTOR_API_KEY environment variable not set")
            cached_models = ["API key not set"]
            return cached_models

        # Create OpenAI client with the Conductor API base URL
        client = openai.OpenAI(base_url=CONDUCTOR_API_BASE, api_key=api_key)

        # Get models using the OpenAI client
        try:
            models = client.models.list()

            # Log the response data
            logging.debug(f"Models response: {models}")

            # Extract model IDs from the response and cache them
            cached_models = [model.id for model in models.data]
            return cached_models
        except Exception as e:
            logging.error(f"Error listing models with OpenAI client: {e}")
            # If we can't get real models, provide some default models for testing
            logging.info("Using default model list for testing")
            cached_models = ["Error: No models found"]
            return cached_models

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        # Return default models as fallback
        cached_models = ["Error: No models found"]
        return cached_models


def query_model(model_name: str, prompt: str) -> Dict[str, Any]:
    """
    Query a specific model with the given prompt.

    Args:
        model_name (str): The name of the model to query
        prompt (str): The prompt to send to the model

    Returns:
        Dict[str, Any]: Dictionary containing response content, tokens, and time
    """
    try:
        # Get API key from environment variable
        api_key = os.environ.get("CONDUCTOR_API_KEY")
        if not api_key:
            return {
                "content": "Error: CONDUCTOR_API_KEY environment variable not set",
                "tokens": 0,
                "time": 0,
            }

        # Create OpenAI client with the Conductor API base URL
        client = openai.OpenAI(base_url=CONDUCTOR_API_BASE, api_key=api_key)

        # Query the model and measure time
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            end_time = time.time()

            # Calculate inference time
            inference_time = end_time - start_time

            # Get token count from response
            completion_tokens = response.usage.completion_tokens

            return {
                "content": response.choices[0].message.content,
                "tokens": completion_tokens,
                "time": round(inference_time, 2),
            }
        except Exception as e:
            logging.error(f"Error querying model: {e}")
            return {"content": f"Error querying model: {e}", "tokens": 0, "time": 0}

    except Exception as e:
        logging.error(f"Error setting up client: {e}")
        return {"content": f"Error setting up client: {e}", "tokens": 0, "time": 0}


def ab_test(model_a: str, model_b: str, query: str) -> List[str]:
    """
    Perform A/B testing between two models.

    Args:
        model_a (str): The name of the first model
        model_b (str): The name of the second model
        query (str): The prompt to send to both models

    Returns:
        List[str]: List containing responses and metrics from both models
    """
    if not query.strip():
        return ["Please enter a query", "Please enter a query", "", "", "", ""]

    # Generate a unique request ID
    global current_request

    # Query both models with the same prompt in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit both queries to the executor
        future_a = executor.submit(query_model, model_a, query)
        future_b = executor.submit(query_model, model_b, query)

        # Get results as they complete
        response_a = future_a.result()
        response_b = future_b.result()

    # Store current request details including responses
    current_request = {
        "id": str(uuid.uuid4()),
        "prompt": query,
        "model_a": model_a,
        "model_b": model_b,
        "response_a": response_a["content"],
        "response_b": response_b["content"],
        "metrics": None,  # Initialize metrics as None
    }

    logging.info(f"New request: {current_request['id']}")

    # Format metrics strings
    metrics_a_str = f"Time: {response_a['time']}s | Tokens: {response_a['tokens']}"
    metrics_b_str = f"Time: {response_b['time']}s | Tokens: {response_b['tokens']}"

    # Calculate similarity metrics between responses
    text_a = response_a["content"]
    text_b = response_b["content"]

    # Only calculate if both responses have content
    if text_a and text_b:
        # Calculate all similarity metrics
        metrics = compute_similarity_metrics(text_a, text_b)

        # Store metrics in current_request for later use
        current_request["metrics"] = metrics

        # Format metrics for display
        similarity_metrics = format_metrics_output(metrics)

        # Generate metrics summary
        metrics_summary = generate_metrics_summary(metrics, text_a, text_b)
    else:
        similarity_metrics = "Cannot calculate similarity (empty response)"
        metrics_summary = ""
        current_request["metrics"] = {
            "jaccard": 0.0,
            "cosine": 0.0,
            "levenshtein": 0.0,
            "semantic": 0.0,
        }

    # Sanitize responses before returning
    sanitized_response_a = sanitize_response(response_a["content"])
    sanitized_response_b = sanitize_response(response_b["content"])

    return [
        sanitized_response_a,
        sanitized_response_b,
        metrics_a_str,
        metrics_b_str,
        similarity_metrics,
        metrics_summary,
    ]


def save_feedback(model_choice: str) -> str:
    """
    Save user feedback to a JSON file.

    Args:
        model_choice (str): The model chosen by the user ('model_a' or 'model_b')

    Returns:
        str: Status message
    """
    global current_request

    # Check if we have a current request
    if not current_request["id"]:
        logging.warning("No active request to save feedback for")
        return "No active request to provide feedback for"

    # Use the pre-computed metrics from current_request
    similarity_metrics = current_request.get(
        "metrics", {"jaccard": 0.0, "cosine": 0.0, "levenshtein": 0.0, "semantic": 0.0}
    )

    # Create feedback entry with similarity metrics
    feedback_entry = {
        "request_id": current_request["id"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_a": current_request["model_a"],
        "model_b": current_request["model_b"],
        "prompt": current_request["prompt"],
        "response_a": current_request["response_a"],
        "response_b": current_request["response_b"],
        "user_choice": model_choice,
        "similarity_metrics": similarity_metrics,
    }

    # Log the metrics being saved
    logging.info(f"Saving feedback with similarity metrics: {similarity_metrics}")

    # Load existing feedback if file exists
    feedback_data = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                feedback_data = json.load(f)
                logging.debug(f"Loaded {len(feedback_data)} existing feedback entries")
        except json.JSONDecodeError:
            logging.error(f"Error reading {FEEDBACK_FILE}, creating new file")

    # Append new feedback
    feedback_data.append(feedback_entry)

    # Save updated feedback
    try:
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback_data, f, indent=2)

        # Verify file was written
        if os.path.exists(FEEDBACK_FILE):
            file_size = os.path.getsize(FEEDBACK_FILE)
            logging.info(
                f"Feedback saved: {feedback_entry['request_id']}. "
                f"File size: {file_size} bytes"
            )
            return f"Feedback saved for {model_choice}"
        else:
            logging.error("File does not exist after write attempt")
            return "Error: File not created"
    except Exception as e:
        logging.error(f"Error saving feedback: {e}")
        return f"Error saving feedback: {e}"


def display_feedback_data() -> str:
    """
    Read and format the feedback data from the JSON file.

    Returns:
        str: Formatted feedback data or error message
    """
    if not os.path.exists(FEEDBACK_FILE):
        return "No feedback data available yet."

    try:
        with open(FEEDBACK_FILE, "r") as f:
            feedback_data = json.load(f)

        if not feedback_data:
            return "Feedback file exists but contains no entries."

        # Format the feedback data for display
        formatted_data = json.dumps(feedback_data, indent=2)
        return formatted_data

    except Exception as e:
        logging.error(f"Error reading feedback file: {e}")
        return f"Error reading feedback file: {e}"


def reset_feedback_file() -> str:
    """
    Reset (clear) the feedback JSON file.

    Returns:
        str: Status message
    """
    try:
        # Create an empty JSON file
        with open(FEEDBACK_FILE, "w") as f:
            json.dump([], f)

        logging.info("Feedback file reset")
        return (
            "Feedback file has been reset. All previous feedback data has been cleared."
        )

    except Exception as e:
        logging.error(f"Error resetting feedback file: {e}")
        return f"Error resetting feedback file: {e}"


def copy_to_clipboard(feedback_data: str) -> str:
    """
    Function to handle the copy to clipboard action.
    This function doesn't actually copy to clipboard (as that happens in JavaScript),
    but returns a status message.

    Args:
        feedback_data (str): The feedback data to be copied

    Returns:
        str: Status message
    """
    if not feedback_data or feedback_data.startswith("No feedback"):
        return "No feedback data to copy."

    return "Feedback data copied to clipboard!"


def clear_outputs() -> List[Any]:
    """
    Clear all outputs and reset model choices.

    Returns:
        List[Any]: Empty values for all outputs and reset model choices
    """
    # Get the first two models for default values
    models = get_available_models()
    default_model_a = models[0] if models else None
    default_model_b = models[1] if len(models) > 1 else models[0] if models else None

    # Reset current request
    global current_request
    current_request = {
        "id": None,
        "prompt": None,
        "model_a": None,
        "model_b": None,
        "response_a": None,
        "response_b": None,
        "metrics": None,
    }

    # Return empty values for all outputs
    # Make sure to return exactly 13 values to match all output components (added one for copy status)
    return [
        "",  # output_a (now Markdown)
        "",  # output_b (now Markdown)
        "",  # query_input
        "",  # metrics_a
        "",  # metrics_b
        "",  # similarity_metrics
        "",  # metrics_summary
        default_model_a,  # model_a_dropdown
        default_model_b,  # model_b_dropdown
        "",  # feedback_status_a
        "",  # feedback_status_b
        "",  # feedback_data_display
        "",  # copy_status
    ]


def load_random_prompt() -> str:
    """
    Load a random prompt from the test_prompts.json file.

    Returns:
        str: A randomly selected prompt or error message
    """
    try:
        # Check if the file exists
        if not os.path.exists("test_prompts.json"):
            logging.warning("test_prompts.json file not found")
            return "Error: test_prompts.json file not found. Please create this file with sample prompts."

        # Load prompts from the file
        with open("test_prompts.json", "r") as f:
            prompts_data = json.load(f)

        # Check if the file contains prompts
        if not prompts_data or not isinstance(prompts_data, list):
            logging.warning("No prompts found in test_prompts.json")
            return "Error: No prompts found in test_prompts.json. Please add some prompts to the file."

        # Select a random prompt
        random_prompt = random.choice(prompts_data)

        # If the prompt is a dictionary with a 'text' field, extract the text
        if isinstance(random_prompt, dict) and "text" in random_prompt:
            return random_prompt["text"]

        # If the prompt is a string, return it directly
        if isinstance(random_prompt, str):
            return random_prompt

        # If the format is unexpected, return an error
        return "Error: Unexpected format in test_prompts.json. Prompts should be strings or dictionaries with a 'text' field."

    except Exception as e:
        logging.error(f"Error loading random prompt: {e}")
        return f"Error loading random prompt: {e}"


def sanitize_response(response_text: str) -> str:
    """
    Sanitize and prepare model response for display in Markdown component.

    Args:
        response_text (str): The raw response text from the model

    Returns:
        str: Sanitized text ready for Markdown display
    """
    if not response_text:
        return "No response received from the model."

    # Check if the response is already in Markdown format
    # If it contains code blocks, ensure they're properly formatted
    if "```" in response_text:
        # Make sure code blocks have language specifiers
        response_text = response_text.replace("```\n", "```text\n")

    # Escape any HTML tags that might interfere with rendering
    response_text = response_text.replace("<", "&lt;").replace(">", "&gt;")

    # Ensure the response starts with a paragraph tag for proper rendering
    if not response_text.startswith("<p>") and not response_text.startswith("#"):
        response_text = f"<p>{response_text}</p>"

    return response_text


def filter_model_reasoning(text: str) -> str:
    """
    Filter out <think></think> sections from model responses.

    Args:
        text (str): The raw response text from the model

    Returns:
        str: Text with <think></think> sections removed
    """
    if not text:
        return text

    # Remove <think>...</think> sections
    filtered_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Clean up any extra whitespace
    filtered_text = re.sub(
        r"\n\s*\n", "\n\n", filtered_text
    )  # Collapse multiple newlines
    filtered_text = filtered_text.strip()

    # If the filtered text is empty, return the original text
    if not filtered_text.strip():
        return text.strip()

    return filtered_text


def compute_similarity_metrics(text1: str, text2: str, model=None) -> Dict[str, float]:
    """
    Compute all similarity metrics between two texts.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        model: Optional pre-loaded SentenceTransformer model for semantic similarity

    Returns:
        Dict[str, float]: Dictionary containing all similarity metrics
    """
    if not text1 or not text2:
        return {"jaccard": 0.0, "cosine": 0.0, "levenshtein": 0.0, "semantic": 0.0}

    # Filter out thinking sections
    text1_filtered = filter_model_reasoning(text1)
    text2_filtered = filter_model_reasoning(text2)

    # Get the model if not provided
    if model is None:
        model = get_model()

    # Calculate all metrics
    # Note: Semantic similarity is computationally expensive and should be calculated only once.
    # The results are stored in current_request["metrics"] to avoid recalculation.
    metrics = {
        "jaccard": jaccard_similarity(text1_filtered, text2_filtered),
        "cosine": cosine_similarity(text1_filtered, text2_filtered),
        "levenshtein": levenshtein_similarity(text1_filtered, text2_filtered),
        "semantic": semantic_similarity(text1_filtered, text2_filtered, model),
    }

    return metrics


def format_metrics_output(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary into a human-readable string with descriptions.

    Args:
        metrics (Dict[str, float]): Dictionary of similarity metrics

    Returns:
        str: Formatted string with metrics and descriptions
    """
    return (
        f"📊 Jaccard: {metrics['jaccard']:.4f} - Measures word overlap between responses\n"
        f"📊 Cosine: {metrics['cosine']:.4f} - Compares word frequency distributions\n"
        f"📊 Levenshtein: {metrics['levenshtein']:.4f} - Calculates character-level edit distance\n"
        f"📊 Semantic: {metrics['semantic']:.4f} - Evaluates meaning similarity using embeddings"
    )


def generate_metrics_summary(metrics: Dict[str, float], text1: str, text2: str) -> str:
    """
    Generate a summary of the similarity metrics using the Blitz model.

    Args:
        metrics (Dict[str, float]): Dictionary of similarity metrics
        text1 (str): First text that was compared
        text2 (str): Second text that was compared

    Returns:
        str: Generated summary of the metrics
    """
    summary_prompt = f"""
    Analyze these similarity metrics between two AI model responses (text A and text B):
    - Jaccard similarity: {metrics['jaccard']:.4f}
    - Cosine similarity: {metrics['cosine']:.4f}
    - Levenshtein similarity: {metrics['levenshtein']:.4f}
    - Semantic similarity: {metrics['semantic']:.4f}
    
    Write a one-paragraph summary explaining what these metrics indicate about the 
    similarity between the responses. You can compare the two answers below. 
    If you see strong similarities or differences, mention them in the summary.
    Keep it concise and informative. Use plain text.

    text A: {text1}
    text B: {text2}
    """

    try:
        summary_response = query_model("virtuoso-large", summary_prompt)
        return summary_response["content"]
    except Exception as e:
        logging.error(f"Error generating metrics summary: {e}")
        return "Could not generate summary of metrics."


# Launch the application when run directly
if __name__ == "__main__":
    print("\nStarting A/B Testing Application...")
    print("============================\n")

    # Get available models
    available_models = get_available_models()
    logging.info(f"Available models: {available_models}")

    # Import and launch the UI
    from ui import create_ui

    demo = create_ui(models_list=available_models)

    # Make the demo object accessible to Gradio's static analysis
    if __name__ == "__main__":
        demo.launch()
