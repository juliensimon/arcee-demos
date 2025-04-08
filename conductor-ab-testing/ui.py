"""
UI module for A/B Testing Application for Conductor API Models

This module contains the Gradio interface for the A/B testing application.
"""

import os
import json
import logging
import gradio as gr
from typing import List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)


# Create the Gradio interface
def create_ui(models_list: List[str] = None):
    # Import functions from app.py here to avoid circular imports
    from app import (
        ab_test,
        save_feedback,
        display_feedback_data,
        reset_feedback_file,
        copy_to_clipboard,
        clear_outputs,
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
        logging.info(f"Available models (refreshed): {available_models}")

    # Create the Gradio interface
    demo = gr.Blocks(
        title="A/B Testing - Arcee Conductor Models",
        css="ui.css",
    )

    with demo:
        # Application title and description
        gr.Markdown("# A/B Testing - Arcee Conductor Models")
        gr.Markdown("Select two models and enter a query (or pick a random prompt) to compare their responses.")

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
            lines=5, placeholder="Enter your query here...", label="Query"
        )

        # Random prompt button with full width
        random_prompt_btn = gr.Button(
            "🎲 Random Prompt",
            variant="secondary",
            elem_classes=["full-width-button"],  # Add a custom class for styling
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
            fn=ab_test,
            inputs=[model_a_dropdown, model_b_dropdown, query_input],
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
            fn=clear_outputs,
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
            ],
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
                    // Create a temporary textarea element
                    const textarea = document.createElement('textarea');
                    textarea.value = feedback_data;
                    textarea.setAttribute('readonly', '');
                    textarea.style.position = 'absolute';
                    textarea.style.left = '-9999px';
                    document.body.appendChild(textarea);
                    
                    // Select and copy the text
                    textarea.select();
                    document.execCommand('copy');
                    
                    // Remove the temporary element
                    document.body.removeChild(textarea);
                    
                    return "✅ Feedback data copied to clipboard!";
                } catch (err) {
                    console.error('Failed to copy: ', err);
                    return "❌ Failed to copy to clipboard. Please try again.";
                }
            }
            """,
        )

        # Reset feedback file button handler
        reset_btn.click(
            fn=reset_feedback_file,  # Call the reset_feedback_file function
            inputs=[],
            outputs=[file_op_status],  # Show the reset status message
        ).then(
            fn=lambda: ("", "", ""),  # Clear feedback status messages and copy status
            inputs=[],
            outputs=[feedback_status_a, feedback_status_b, copy_status],
        ).then(
            fn=display_feedback_data,  # Update the feedback display
            inputs=[],
            outputs=[feedback_data_display],
        )

        # Random prompt button handler
        random_prompt_btn.click(fn=load_random_prompt, inputs=[], outputs=[query_input])

    return demo
