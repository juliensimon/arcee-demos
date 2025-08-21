"""
UI module for A/B Testing Application

This module contains the Gradio interface for the A/B testing application,
providing a user-friendly web interface for comparing AI model responses
with configurable generation parameters and comprehensive analysis tools.

Features:
- Side-by-side model comparison interface
- Configurable generation parameters (temperature, top_p, max_tokens, etc.)
- Real-time similarity metrics display
- User feedback collection system
- Random prompt loading for testing
- Comprehensive data management tools

Author: Arcee AI Team
License: MIT
"""

import os
import json
import logging
import gradio as gr
from typing import List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_generation_params_dict(
    temperature: float, 
    top_p: float, 
    max_tokens: int, 
    frequency_penalty: float, 
    presence_penalty: float
) -> dict:
    """
    Create a dictionary of generation parameters from UI inputs.
    
    This function converts the individual UI parameter values into a structured
    dictionary that can be passed to the model querying functions.
    
    Args:
        temperature (float): Temperature parameter for controlling randomness.
        top_p (float): Top P parameter for nucleus sampling.
        max_tokens (int): Maximum number of tokens to generate.
        frequency_penalty (float): Penalty for frequent token repetition.
        presence_penalty (float): Penalty for any token repetition.
        
    Returns:
        dict: Dictionary containing all generation parameters with proper types.
    
    Note:
        All parameters are explicitly converted to their expected types to ensure
        compatibility with the model API calls.
    """
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "frequency_penalty": float(frequency_penalty),
        "presence_penalty": float(presence_penalty),
    }


def create_ui(models_list: Optional[List[str]] = None) -> gr.Blocks:
    """
    Create the Gradio interface for the A/B testing application.
    
    This function builds a comprehensive web interface that allows users to:
    - Select and compare two different AI models
    - Configure generation parameters for fair comparison
    - View detailed metrics and similarity analysis
    - Collect and manage user feedback
    - Load random prompts for testing
    
    Args:
        models_list (Optional[List[str]]): List of available model names.
            If None, models will be fetched automatically from the app module.
    
    Returns:
        gr.Blocks: Configured Gradio interface ready for launch.
    
    Note:
        The interface includes all necessary event handlers and provides
        a complete workflow for A/B testing AI models with comprehensive
        analysis and feedback collection capabilities.
    """
    # Import functions from app.py here to avoid circular imports
    from app import (
        ab_test,
        save_feedback,
        display_feedback_data,
        reset_feedback_file,
        copy_to_clipboard,
        clear_outputs_with_params,
        load_random_prompt,
        get_available_models,
    )

    # Use provided models list or get a fresh one if None
    available_models = (
        models_list
        if models_list is not None
        else get_available_models(force_refresh=True)
    )

    # Only log if we had to refresh the models (not using the passed list)
    if models_list is None:
        logger.info(f"Available models (refreshed): {len(available_models)} models")

    # Create the Gradio interface
    demo = gr.Blocks(
        title="A/B Testing - Arcee Conductor Models",
        css="ui.css",
    )

    with demo:
        # Application title and description
        gr.Markdown("# A/B Testing - Arcee Conductor Models")
        gr.Markdown(
            "Select two models and enter a query (or pick a random prompt) to compare their responses. "
            "Both models will use identical generation parameters for fair comparison."
        )

        # Model selection dropdowns
        with gr.Row():
            model_a_dropdown = gr.Dropdown(
                choices=available_models,
                label="Model A",
                value=available_models[0] if available_models else None,
            )
            model_b_dropdown = gr.Dropdown(
                choices=available_models,
                label="Model B",
                value=(
                    available_models[1]
                    if len(available_models) > 1
                    else available_models[0] if available_models else None
                ),
            )

        # Query input box
        query_input = gr.Textbox(
            lines=5, 
            placeholder="Enter your query here...", 
            label="Query"
        )

        # Generation Parameters Section
        with gr.Accordion("🔧 Generation Parameters", open=False):
            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness (0.0 = deterministic, 2.0 = very random)"
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="Top P",
                        info="Controls diversity via nucleus sampling"
                    )
                with gr.Column():
                    max_tokens = gr.Slider(
                        minimum=1,
                        maximum=4000,
                        value=1000,
                        step=1,
                        label="Max Tokens",
                        info="Maximum number of tokens to generate"
                    )
                    frequency_penalty = gr.Slider(
                        minimum=-2.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        label="Frequency Penalty",
                        info="Reduces repetition of frequent tokens"
                    )
                    presence_penalty = gr.Slider(
                        minimum=-2.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        label="Presence Penalty",
                        info="Reduces repetition of any token"
                    )

        # Random prompt button with full width
        random_prompt_btn = gr.Button(
            "🎲 Random Prompt",
            variant="secondary",
            elem_classes=["full-width-button"],
        )

        # Submit and clear buttons
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")

        # Model responses and metrics display
        with gr.Row():
            # Model A column
            with gr.Column():
                gr.Markdown("### Model A Response")
                output_a = gr.Markdown(elem_id="output-a-markdown")
                metrics_a = gr.Textbox(label="Metrics", lines=1)

                # Add feedback button for Model A
                with gr.Row(elem_classes=["feedback-container"]):
                    thumbs_up_a = gr.Button(
                        "👍 Prefer this response",
                        size="sm",
                        elem_classes=["thumbs-button"],
                    )

                # Display feedback status for Model A
                feedback_status_a = gr.Textbox(
                    label="", value="", elem_classes=["feedback-status"]
                )

            # Model B column
            with gr.Column():
                gr.Markdown("### Model B Response")
                output_b = gr.Markdown(elem_id="output-b-markdown")
                metrics_b = gr.Textbox(label="Metrics", lines=1)

                # Add feedback button for Model B
                with gr.Row(elem_classes=["feedback-container"]):
                    thumbs_up_b = gr.Button(
                        "👍 Prefer this response",
                        size="sm",
                        elem_classes=["thumbs-button"],
                    )

                # Display feedback status for Model B
                feedback_status_b = gr.Textbox(
                    label="", value="", elem_classes=["feedback-status"]
                )

        # Similarity metrics section
        gr.Markdown("### Similarity Metrics", elem_classes=["center-text"])

        # Display similarity metrics and summary side by side
        with gr.Row():
            with gr.Column():
                similarity_metrics = gr.Textbox(
                    label="", lines=4, elem_id="similarity-metrics"
                )

            with gr.Column():
                metrics_summary = gr.Textbox(
                    label="", lines=4, elem_id="metrics-summary"
                )

        # Feedback data management section
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "### Feedback Data Management", elem_classes=["center-text"]
                )

                # Add file action buttons
                with gr.Row(elem_classes=["file-actions"]):
                    display_btn = gr.Button(
                        "Display Feedback Data", variant="secondary"
                    )
                    copy_btn = gr.Button(
                        "📋 Copy to Clipboard", elem_classes=["copy-btn"]
                    )
                    reset_btn = gr.Button(
                        "Reset Feedback File", elem_classes=["warning-btn"]
                    )

                # Display feedback data
                feedback_data_display = gr.Textbox(
                    label="", lines=10, elem_id="feedback-data"
                )

                # Display status message for file operations
                file_op_status = gr.Textbox(
                    label="", value="", elem_classes=["feedback-status"]
                )

                # Add a hidden component to show copy status
                copy_status = gr.Textbox(
                    label="", value="", elem_classes=["feedback-status"], visible=True
                )

        # Set up event handlers
        # Submit button handler
        submit_btn.click(
            fn=lambda model_a, model_b, query, temp, top_p, max_tok, freq_pen, pres_pen: ab_test(
                model_a, model_b, query, create_generation_params_dict(temp, top_p, max_tok, freq_pen, pres_pen)
            ),
            inputs=[
                model_a_dropdown, 
                model_b_dropdown, 
                query_input,
                temperature,
                top_p,
                max_tokens,
                frequency_penalty,
                presence_penalty
            ],
            outputs=[
                output_a,
                output_b,
                metrics_a,
                metrics_b,
                similarity_metrics,
                metrics_summary,
            ],
        )

        # Clear button handler
        clear_btn.click(
            fn=clear_outputs_with_params,
            inputs=[],
            outputs=[
                output_a,
                output_b,
                query_input,
                metrics_a,
                metrics_b,
                similarity_metrics,
                metrics_summary,
                model_a_dropdown,
                model_b_dropdown,
                feedback_status_a,
                feedback_status_b,
                feedback_data_display,
                copy_status,
                temperature,
                top_p,
                max_tokens,
                frequency_penalty,
                presence_penalty,
            ],
        )

        # Random prompt button handler
        random_prompt_btn.click(
            fn=load_random_prompt,
            inputs=[],
            outputs=[query_input],
        )

        # Model A feedback button handler
        thumbs_up_a.click(
            fn=lambda: save_feedback("model_a"), inputs=[], outputs=[feedback_status_a]
        ).then(fn=display_feedback_data, inputs=[], outputs=[feedback_data_display])

        # Model B feedback button handler
        thumbs_up_b.click(
            fn=lambda: save_feedback("model_b"), inputs=[], outputs=[feedback_status_b]
        ).then(fn=display_feedback_data, inputs=[], outputs=[feedback_data_display])

        # Display feedback data button handler
        display_btn.click(
            fn=display_feedback_data, inputs=[], outputs=[feedback_data_display]
        )

        # Copy to clipboard button handler
        copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[feedback_data_display],
            outputs=[copy_status],
            js="""
            function(feedback_data) {
                if (!feedback_data || feedback_data.startsWith("No feedback")) {
                    return "No feedback data to copy.";
                }
               
                // Copy to clipboard using a more reliable approach
                try {
                    const textArea = document.createElement('textarea');
                    textArea.value = feedback_data;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    return "Feedback data copied to clipboard!";
                } catch (err) {
                    console.error('Failed to copy: ', err);
                    return "Failed to copy to clipboard.";
                }
            }
            """,
        )

        # Reset feedback file button handler
        reset_btn.click(
            fn=reset_feedback_file,
            inputs=[],
            outputs=[file_op_status],
        ).then(fn=display_feedback_data, inputs=[], outputs=[feedback_data_display])

    logger.info("Gradio interface created successfully")
    return demo
