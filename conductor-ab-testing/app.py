"""
A/B Testing Application for Conductor API Models

This application allows users to compare responses from different AI models
through the Conductor API and Together.ai, collect feedback, and analyze 
similarity metrics with configurable generation parameters.

Features:
- Multi-provider model support (Conductor API, Together.ai)
- Configurable generation parameters (temperature, top_p, max_tokens, etc.)
- Side-by-side model comparison with detailed metrics
- Semantic similarity analysis
- User feedback collection and management
- Random prompt loading for testing

Author: Arcee AI Team
License: MIT
"""

# Standard library imports
import os
import time
import json
import uuid
import logging
import random
import concurrent.futures
import re
from typing import List, Dict, Any, Tuple, Optional

# Third-party imports
import openai
from together import Together

# Local imports
from similarity import (
    jaccard_similarity,
    cosine_similarity,
    levenshtein_similarity,
    semantic_similarity,
)
from model_loader import get_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CONDUCTOR_API_BASE = "https://conductor.arcee.ai/v1"
FEEDBACK_FILE = "user_feedback.json"

# Global state
current_request: Dict[str, Any] = {
    "id": None,
    "prompt": None,
    "model_a": None,
    "model_b": None,
    "response_a": None,
    "response_b": None,
    "metrics": None,
}

cached_models: Optional[List[str]] = None


def get_default_generation_params() -> Dict[str, Any]:
    """
    Get default generation parameters for model inference.
    
    These parameters provide a balanced configuration for most use cases:
    - Moderate creativity with temperature 0.7
    - Good diversity with top_p 0.9
    - Reasonable response length with max_tokens 1000
    - No repetition penalties by default
    
    Returns:
        Dict[str, Any]: Dictionary containing default generation parameters
            - temperature (float): Controls randomness (0.0-2.0)
            - top_p (float): Controls diversity via nucleus sampling (0.0-1.0)
            - max_tokens (int): Maximum tokens to generate (1-4000)
            - frequency_penalty (float): Reduces repetition of frequent tokens (-2.0-2.0)
            - presence_penalty (float): Reduces repetition of any token (-2.0-2.0)
    """
    return {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1000,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }


def get_available_models(force_refresh: bool = False) -> List[str]:
    """
    Get a list of available models from both Conductor API and Together.ai.
    
    This function aggregates models from multiple providers and caches the results
    to avoid repeated API calls. It handles API errors gracefully and provides
    informative logging for debugging.
    
    Args:
        force_refresh (bool): If True, forces a refresh of the cached models.
            Useful when new models become available or API keys are updated.
    
    Returns:
        List[str]: List of available model names. May include error messages
            prefixed with "Error:" if certain providers are unavailable.
    
    Note:
        Models are cached globally to improve performance. Use force_refresh=True
        to bypass the cache when needed.
    """
    global cached_models

    # Return cached models if available and not forcing refresh
    if cached_models is not None and not force_refresh:
        logger.debug("Returning cached models")
        return cached_models

    models_list = []
    
    # Add Together.ai models
    together_api_key = os.environ.get("TOGETHER_API_KEY")
    if together_api_key:
        models_list.append("arcee-ai/AFM-4.5B")
        logger.info("Added AFM-4.5B model from Together.ai")
    else:
        logger.warning("TOGETHER_API_KEY environment variable not set - AFM-4.5B not available")

    # Get Conductor API models
    conductor_api_key = os.environ.get("CONDUCTOR_API_KEY")
    if conductor_api_key:
        try:
            # Create OpenAI client with the Conductor API base URL
            client = openai.OpenAI(base_url=CONDUCTOR_API_BASE, api_key=conductor_api_key)

            # Get models using the OpenAI client
            try:
                models = client.models.list()
                logger.debug(f"Retrieved {len(models.data)} models from Conductor API")

                # Extract model IDs from the response
                conductor_models = [model.id for model in models.data]
                models_list.extend(conductor_models)
                logger.info(f"Added {len(conductor_models)} Conductor API models")
                
            except Exception as e:
                logger.error(f"Error listing models with OpenAI client: {e}")
                models_list.append("Error: No Conductor models found")
        except Exception as e:
            logger.error(f"Unexpected error with Conductor API: {e}")
            models_list.append("Error: Conductor API error")
    else:
        logger.warning("CONDUCTOR_API_KEY environment variable not set - Conductor models not available")
        models_list.append("Error: CONDUCTOR_API_KEY not set")

    # Cache the combined models list
    cached_models = models_list
    logger.info(f"Total models available: {len(models_list)}")
    return cached_models


def get_model_client(model_name: str) -> Tuple[Any, bool]:
    """
    Get the appropriate API client for a given model.
    
    This function determines which API provider to use based on the model name
    and returns the corresponding client along with a flag indicating the provider.
    
    Args:
        model_name (str): The name of the model to get a client for.
            Currently supports "arcee-ai/AFM-4.5B" for Together.ai and
            all other models for Conductor API.
    
    Returns:
        Tuple[Any, bool]: A tuple containing:
            - client: The API client instance (Together or OpenAI client)
            - is_together_model: Boolean indicating if this is a Together.ai model
    
    Raises:
        ValueError: If the required API key is not set in environment variables.
    
    Note:
        The function automatically detects Together.ai models by checking for
        the "arcee-ai/AFM-4.5B" model name. All other models are assumed to be
        Conductor API models.
    """
    # Check if this is a Together.ai model
    if model_name == "arcee-ai/AFM-4.5B":
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        logger.debug(f"Using Together.ai client for model: {model_name}")
        return Together(api_key=api_key), True
    
    # Default to Conductor API
    api_key = os.environ.get("CONDUCTOR_API_KEY")
    if not api_key:
        raise ValueError("CONDUCTOR_API_KEY environment variable not set")
    logger.debug(f"Using Conductor API client for model: {model_name}")
    return openai.OpenAI(base_url=CONDUCTOR_API_BASE, api_key=api_key), False


def query_model(model_name: str, prompt: str, generation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Query a specific model with the given prompt and generation parameters.
    
    This function handles model inference across different API providers (Conductor API
    and Together.ai) with consistent parameter handling and error management.
    
    Args:
        model_name (str): The name of the model to query.
        prompt (str): The prompt to send to the model.
        generation_params (Optional[Dict[str, Any]]): Generation parameters for the model.
            If None, default parameters will be used. Supported parameters:
            - temperature (float): Controls randomness (0.0-2.0)
            - top_p (float): Controls diversity via nucleus sampling (0.0-1.0)
            - max_tokens (int): Maximum tokens to generate (1-4000)
            - frequency_penalty (float): Reduces repetition of frequent tokens (-2.0-2.0)
            - presence_penalty (float): Reduces repetition of any token (-2.0-2.0)
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - content (str): The model's response text
            - tokens (int): Number of tokens generated
            - time (float): Inference time in seconds (rounded to 2 decimal places)
    
    Note:
        The function automatically handles different API providers based on the model name.
        Both providers use the same parameter structure for consistency.
    """
    # Set default generation parameters if none provided
    if generation_params is None:
        generation_params = get_default_generation_params()
    
    logger.debug(f"Querying model '{model_name}' with parameters: {generation_params}")
    
    try:
        # Get the appropriate client for this model
        client, is_together_model = get_model_client(model_name)

        # Query the model and measure time
        start_time = time.time()
        try:
            # Prepare the request parameters
            request_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                **generation_params
            }
            
            # Create the response using the same parameters for both providers
            response = client.chat.completions.create(**request_params)
            
            end_time = time.time()

            # Calculate inference time
            inference_time = end_time - start_time

            # Get token count from response
            completion_tokens = response.usage.completion_tokens

            result = {
                "content": response.choices[0].message.content,
                "tokens": completion_tokens,
                "time": round(inference_time, 2),
            }
            
            logger.debug(f"Model '{model_name}' response: {completion_tokens} tokens in {result['time']}s")
            return result
            
        except Exception as e:
            logger.error(f"Error querying model '{model_name}': {e}")
            return {"content": f"Error querying model: {e}", "tokens": 0, "time": 0}

    except Exception as e:
        logger.error(f"Error setting up client for model '{model_name}': {e}")
        return {"content": f"Error setting up client: {e}", "tokens": 0, "time": 0}


def ab_test(model_a: str, model_b: str, query: str, generation_params: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Perform A/B testing between two models with identical parameters.
    
    This function executes parallel queries to two different models using the same
    prompt and generation parameters, then calculates similarity metrics between
    their responses. The results are stored globally for feedback collection.
    
    Args:
        model_a (str): The name of the first model to test.
        model_b (str): The name of the second model to test.
        query (str): The prompt to send to both models.
        generation_params (Optional[Dict[str, Any]]): Generation parameters for both models.
            If None, default parameters will be used. Both models use identical parameters
            to ensure fair comparison.
    
    Returns:
        List[str]: A list containing 6 elements in the following order:
            - Model A response (sanitized for display)
            - Model B response (sanitized for display)
            - Model A metrics string (time and tokens)
            - Model B metrics string (time and tokens)
            - Similarity metrics formatted string
            - Metrics summary generated by AI
    
    Note:
        The function uses ThreadPoolExecutor for parallel execution to minimize
        total testing time. Both models receive identical parameters to ensure
        fair comparison. Results are stored in the global current_request for
        feedback collection.
    """
    if not query.strip():
        logger.warning("Empty query provided to A/B test")
        return ["Please enter a query", "Please enter a query", "", "", "", ""]

    # Generate a unique request ID
    global current_request

    logger.info(f"Starting A/B test: {model_a} vs {model_b}")
    logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    logger.debug(f"Generation parameters: {generation_params}")

    # Query both models with the same prompt and parameters in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit both queries to the executor with the same generation parameters
        future_a = executor.submit(query_model, model_a, query, generation_params)
        future_b = executor.submit(query_model, model_b, query, generation_params)

        # Get results as they complete
        response_a = future_a.result()
        response_b = future_b.result()

    # Store current request details including responses
    request_id = str(uuid.uuid4())
    current_request = {
        "id": request_id,
        "prompt": query,
        "model_a": model_a,
        "model_b": model_b,
        "response_a": response_a["content"],
        "response_b": response_b["content"],
        "metrics": None,  # Initialize metrics as None
    }

    logger.info(f"New request completed: {request_id}")

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
        metrics_summary = generate_metrics_summary(metrics, text_a, text_b, generation_params)
        
        logger.debug(f"Similarity metrics calculated: {metrics}")
    else:
        similarity_metrics = "Cannot calculate similarity (empty response)"
        metrics_summary = ""
        current_request["metrics"] = {
            "jaccard": 0.0,
            "cosine": 0.0,
            "levenshtein": 0.0,
            "semantic": 0.0,
        }
        logger.warning("One or both responses were empty, cannot calculate similarity")

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
    Save user feedback to a JSON file with comprehensive metadata.
    
    This function saves user preferences along with the complete test context,
    including model responses, similarity metrics, and test parameters. The data
    is stored in a structured JSON format for later analysis.
    
    Args:
        model_choice (str): The model chosen by the user. Must be either 'model_a'
            or 'model_b' to indicate which model's response was preferred.
    
    Returns:
        str: Status message indicating success or failure of the save operation.
    
    Note:
        The function requires a valid current_request in the global state.
        If no active request exists, it returns an error message.
        The feedback file is automatically created if it doesn't exist.
    """
    global current_request

    # Validate model choice
    if model_choice not in ['model_a', 'model_b']:
        logger.error(f"Invalid model choice: {model_choice}")
        return f"Error: Invalid model choice '{model_choice}'"

    # Check if we have a current request
    if not current_request["id"]:
        logger.warning("No active request to save feedback for")
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

    logger.info(f"Saving feedback for {model_choice} with metrics: {similarity_metrics}")

    # Load existing feedback if file exists
    feedback_data = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding='utf-8') as f:
                feedback_data = json.load(f)
                logger.debug(f"Loaded {len(feedback_data)} existing feedback entries")
        except json.JSONDecodeError as e:
            logger.error(f"Error reading {FEEDBACK_FILE}: {e}, creating new file")
        except Exception as e:
            logger.error(f"Unexpected error reading {FEEDBACK_FILE}: {e}")
            return f"Error reading feedback file: {e}"

    # Append new feedback
    feedback_data.append(feedback_entry)

    # Save updated feedback
    try:
        with open(FEEDBACK_FILE, "w", encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)

        # Verify file was written
        if os.path.exists(FEEDBACK_FILE):
            file_size = os.path.getsize(FEEDBACK_FILE)
            logger.info(
                f"Feedback saved successfully: {feedback_entry['request_id']} "
                f"(file size: {file_size} bytes, total entries: {len(feedback_data)})"
            )
            return f"Feedback saved for {model_choice}"
        else:
            logger.error("File does not exist after write attempt")
            return "Error: File not created"
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return f"Error saving feedback: {e}"


def display_feedback_data() -> str:
    """
    Read and format the feedback data from the JSON file for display.
    
    This function loads the feedback data from the JSON file and formats it
    as a readable JSON string for display in the UI. It handles various error
    conditions gracefully.
    
    Returns:
        str: Formatted JSON string of feedback data, or an error message if
            the file cannot be read or doesn't exist.
    
    Note:
        The function returns a user-friendly message if no feedback data exists,
        and provides detailed error information if file operations fail.
    """
    if not os.path.exists(FEEDBACK_FILE):
        logger.info("No feedback file exists yet")
        return "No feedback data available yet."

    try:
        with open(FEEDBACK_FILE, "r", encoding='utf-8') as f:
            feedback_data = json.load(f)

        if not feedback_data:
            logger.info("Feedback file exists but is empty")
            return "Feedback file exists but contains no entries."

        # Format the feedback data for display
        formatted_data = json.dumps(feedback_data, indent=2, ensure_ascii=False)
        logger.debug(f"Displayed {len(feedback_data)} feedback entries")
        return formatted_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error reading feedback file: {e}")
        return f"Error: Invalid JSON in feedback file - {e}"
    except Exception as e:
        logger.error(f"Error reading feedback file: {e}")
        return f"Error reading feedback file: {e}"


def reset_feedback_file() -> str:
    """
    Reset (clear) the feedback JSON file.
    
    This function clears all existing feedback data by overwriting the feedback
    file with an empty JSON array. This operation is irreversible.
    
    Returns:
        str: Status message indicating success or failure of the reset operation.
    
    Note:
        This operation permanently deletes all stored feedback data. Use with caution.
        The function creates the feedback file if it doesn't exist.
    """
    try:
        # Create an empty JSON file
        with open(FEEDBACK_FILE, "w", encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)

        logger.info("Feedback file reset successfully")
        return "Feedback file has been reset. All previous feedback data has been cleared."

    except Exception as e:
        logger.error(f"Error resetting feedback file: {e}")
        return f"Error resetting feedback file: {e}"


def copy_to_clipboard(feedback_data: str) -> str:
    """
    Handle the copy to clipboard action for feedback data.
    
    This function validates the feedback data and returns an appropriate status
    message. The actual clipboard copying is handled by JavaScript in the UI.
    
    Args:
        feedback_data (str): The feedback data to be copied to clipboard.
    
    Returns:
        str: Status message indicating whether the data can be copied or not.
    
    Note:
        This function only validates the data. The actual clipboard operation
        is performed by JavaScript in the Gradio interface.
    """
    if not feedback_data or feedback_data.startswith("No feedback"):
        logger.debug("No valid feedback data to copy")
        return "No feedback data to copy."

    logger.debug("Feedback data validated for clipboard copy")
    return "Feedback data copied to clipboard!"


def clear_outputs() -> List[Any]:
    """
    Clear all outputs and reset model choices to defaults.
    
    This function resets the UI state by clearing all response outputs and
    resetting model selections to the first two available models. It also
    clears the global current_request state.
    
    Returns:
        List[Any]: A list of 13 values containing empty strings for outputs
            and default model selections for dropdowns. The order matches
            the UI component outputs exactly.
    
    Note:
        This function is used by the "Clear" button in the UI to reset the
        interface state without affecting generation parameters.
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

    logger.debug("Cleared all outputs and reset model choices")

    # Return empty values for all outputs
    # Make sure to return exactly 13 values to match all output components
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


def clear_outputs_with_params() -> List[Any]:
    """
    Clear all outputs, reset model choices, and reset generation parameters to defaults.
    
    This function performs a complete UI reset including clearing all response outputs,
    resetting model selections to the first two available models, and resetting all
    generation parameters to their default values. It also clears the global
    current_request state.
    
    Returns:
        List[Any]: A list of 18 values containing:
            - 13 empty strings for outputs and default model selections
            - 5 default generation parameter values (temperature, top_p, max_tokens,
              frequency_penalty, presence_penalty)
    
    Note:
        This function is used by the "Clear" button in the UI to perform a complete
        reset of the interface state, including generation parameters.
    """
    # Get the first two models for default values
    models = get_available_models()
    default_model_a = models[0] if models else None
    default_model_b = models[1] if len(models) > 1 else models[0] if models else None

    # Get default generation parameters
    default_params = get_default_generation_params()

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

    logger.debug("Cleared all outputs, reset model choices, and reset generation parameters")

    # Return empty values for all outputs plus default generation parameters
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
        default_params["temperature"],  # temperature
        default_params["top_p"],  # top_p
        default_params["max_tokens"],  # max_tokens
        default_params["frequency_penalty"],  # frequency_penalty
        default_params["presence_penalty"],  # presence_penalty
    ]


def load_random_prompt() -> str:
    """
    Load a random prompt from the test_prompts.json file.
    
    This function loads a random prompt from the test_prompts.json file to help
    users quickly test different models. It supports both string and dictionary
    formats for prompts and provides detailed error messages for troubleshooting.
    
    Returns:
        str: A randomly selected prompt from the file, or an error message if
            the file cannot be loaded or doesn't contain valid prompts.
    
    Note:
        The function expects test_prompts.json to contain either:
        - A list of strings (prompts)
        - A list of dictionaries with a 'text' field
        If the file doesn't exist or has an invalid format, an error message is returned.
    """
    try:
        # Check if the file exists
        if not os.path.exists("test_prompts.json"):
            logger.warning("test_prompts.json file not found")
            return "Error: test_prompts.json file not found. Please create this file with sample prompts."

        # Load prompts from the file
        with open("test_prompts.json", "r", encoding='utf-8') as f:
            prompts_data = json.load(f)

        # Check if the file contains prompts
        if not prompts_data or not isinstance(prompts_data, list):
            logger.warning("No prompts found in test_prompts.json")
            return "Error: No prompts found in test_prompts.json. Please add some prompts to the file."

        # Select a random prompt
        random_prompt = random.choice(prompts_data)

        # If the prompt is a dictionary with a 'text' field, extract the text
        if isinstance(random_prompt, dict) and "text" in random_prompt:
            logger.debug("Loaded random prompt from dictionary format")
            return random_prompt["text"]

        # If the prompt is a string, return it directly
        if isinstance(random_prompt, str):
            logger.debug("Loaded random prompt from string format")
            return random_prompt

        # If the format is unexpected, return an error
        logger.error(f"Unexpected prompt format: {type(random_prompt)}")
        return "Error: Unexpected format in test_prompts.json. Prompts should be strings or dictionaries with a 'text' field."

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error loading test_prompts.json: {e}")
        return f"Error: Invalid JSON in test_prompts.json - {e}"
    except Exception as e:
        logger.error(f"Error loading random prompt: {e}")
        return f"Error loading random prompt: {e}"


def sanitize_response(response_text: str) -> str:
    """
    Sanitize and prepare model response for display in Markdown component.
    
    This function processes raw model responses to ensure they display correctly
    in the Gradio Markdown component. It handles code blocks, HTML escaping,
    and proper formatting for consistent rendering.
    
    Args:
        response_text (str): The raw response text from the model.
    
    Returns:
        str: Sanitized text ready for Markdown display, or a default message
            if no response text is provided.
    
    Note:
        The function performs the following sanitization steps:
        - Adds language specifiers to code blocks if missing
        - Escapes HTML tags to prevent rendering issues
        - Wraps text in paragraph tags if not already formatted
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
    
    This function removes internal reasoning sections from model responses that
    are typically enclosed in <think></think> tags. It also cleans up whitespace
    to improve readability of the final response.
    
    Args:
        text (str): The raw response text from the model, which may contain
            <think></think> sections.
    
    Returns:
        str: Cleaned text with <think></think> sections removed and whitespace
            normalized. If the filtered text is empty, returns the original text.
    
    Note:
        The function uses regex to remove all content between <think> and </think>
        tags, including nested content. It also collapses multiple newlines and
        trims whitespace for better formatting.
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
    
    This function calculates multiple similarity metrics between two text responses
    to provide comprehensive comparison analysis. It filters out internal reasoning
    sections before computing metrics for more accurate comparison.
    
    Args:
        text1 (str): First text to compare.
        text2 (str): Second text to compare.
        model: Optional pre-loaded SentenceTransformer model for semantic similarity.
            If None, the model will be loaded automatically.
    
    Returns:
        Dict[str, float]: Dictionary containing similarity metrics:
            - jaccard (float): Word overlap similarity (0.0-1.0)
            - cosine (float): Word frequency distribution similarity (0.0-1.0)
            - levenshtein (float): Character-level edit distance similarity (0.0-1.0)
            - semantic (float): Meaning similarity using embeddings (0.0-1.0)
    
    Note:
        The function filters out <think></think> sections before computing metrics
        to focus on the actual response content. If either text is empty, all
        metrics return 0.0.
    """
    if not text1 or not text2:
        logger.debug("Empty text provided, returning zero metrics")
        return {"jaccard": 0.0, "cosine": 0.0, "levenshtein": 0.0, "semantic": 0.0}

    # Filter out thinking sections
    text1_filtered = filter_model_reasoning(text1)
    text2_filtered = filter_model_reasoning(text2)

    # Get the model if not provided
    if model is None:
        model = get_model()

    logger.debug("Computing similarity metrics between filtered texts")

    # Calculate all metrics
    # Note: Semantic similarity is computationally expensive and should be calculated only once.
    # The results are stored in current_request["metrics"] to avoid recalculation.
    metrics = {
        "jaccard": jaccard_similarity(text1_filtered, text2_filtered),
        "cosine": cosine_similarity(text1_filtered, text2_filtered),
        "levenshtein": levenshtein_similarity(text1_filtered, text2_filtered),
        "semantic": semantic_similarity(text1_filtered, text2_filtered, model),
    }

    logger.debug(f"Computed similarity metrics: {metrics}")
    return metrics


def format_metrics_output(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary into a human-readable string with descriptions.
    
    This function converts the raw similarity metrics into a user-friendly format
    with emojis, formatted values, and brief descriptions of what each metric measures.
    
    Args:
        metrics (Dict[str, float]): Dictionary of similarity metrics containing
            jaccard, cosine, levenshtein, and semantic similarity scores.
    
    Returns:
        str: Formatted string with each metric on a new line, including:
            - Emoji icon for visual appeal
            - Metric value formatted to 4 decimal places
            - Brief description of what the metric measures
    
    Note:
        The function expects metrics to contain 'jaccard', 'cosine', 'levenshtein',
        and 'semantic' keys. If any are missing, it may raise a KeyError.
    """
    return (
        f"📊 Jaccard: {metrics['jaccard']:.4f} - Measures word overlap between responses\n"
        f"📊 Cosine: {metrics['cosine']:.4f} - Compares word frequency distributions\n"
        f"📊 Levenshtein: {metrics['levenshtein']:.4f} - Calculates character-level edit distance\n"
        f"📊 Semantic: {metrics['semantic']:.4f} - Evaluates meaning similarity using embeddings"
    )


def generate_metrics_summary(metrics: Dict[str, float], text1: str, text2: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a summary of the similarity metrics using an available model.
    
    This function uses an AI model to analyze the similarity metrics between two
    responses and generate a human-readable summary. It intelligently selects
    the best available model for the summary generation.
    
    Args:
        metrics (Dict[str, float]): Dictionary of similarity metrics containing
            jaccard, cosine, levenshtein, and semantic similarity scores.
        text1 (str): First text that was compared (Model A response).
        text2 (str): Second text that was compared (Model B response).
        generation_params (Optional[Dict[str, Any]]): Generation parameters for
            the summary model. If None, default parameters will be used.
    
    Returns:
        str: Generated summary of the metrics, or an error message if summary
            generation fails.
    
    Note:
        The function prioritizes AFM-4.5B for summary generation if available,
        otherwise uses the first available non-error model. The summary provides
        insights into the similarity between the two model responses based on
        the computed metrics.
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
        # Get available models and use the first one that's not an error
        available_models = get_available_models()
        summary_model = None
        
        # Try to use AFM-4.5B first if available
        if "arcee-ai/AFM-4.5B" in available_models:
            summary_model = "arcee-ai/AFM-4.5B"
            logger.debug("Using AFM-4.5B for metrics summary generation")
        else:
            # Use the first available model that's not an error
            for model in available_models:
                if not model.startswith("Error:"):
                    summary_model = model
                    logger.debug(f"Using {model} for metrics summary generation")
                    break
        
        if summary_model:
            logger.debug(f"Generating metrics summary using {summary_model}")
            summary_response = query_model(summary_model, summary_prompt, generation_params)
            return summary_response["content"]
        else:
            logger.warning("No available models for generating summary")
            return "No available models for generating summary."
            
    except Exception as e:
        logger.error(f"Error generating metrics summary: {e}")
        return "Could not generate summary of metrics."


# Launch the application when run directly
if __name__ == "__main__":
    print("\n🚀 Starting A/B Testing Application...")
    print("=" * 50)
    print("📊 Multi-Provider Model Comparison Tool")
    print("🔧 Configurable Generation Parameters")
    print("📈 Comprehensive Similarity Analysis")
    print("=" * 50)

    try:
        # Get available models
        available_models = get_available_models()
        logger.info(f"Available models: {len(available_models)} models loaded")
        
        if available_models:
            logger.info(f"Model providers: {[m for m in available_models if not m.startswith('Error:')]}")
        else:
            logger.warning("No models available - check API keys")

        # Import and launch the UI
        from ui import create_ui

        demo = create_ui(models_list=available_models)
        logger.info("UI created successfully")

        # Launch the application
        demo.launch()
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"❌ Error starting application: {e}")
        raise
