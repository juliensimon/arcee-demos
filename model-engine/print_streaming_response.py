from IPython.display import Markdown, clear_output, display


def print_streaming_response(response):
    num_tokens = 0
    content = ""
    model_id = None

    for message in response:
        if len(message.choices) > 0:
            # Capture model ID from the first chunk if available
            if model_id is None and hasattr(message, "model"):
                model_id = message.model

            num_tokens += 1
            chunk = message.choices[0].delta.content
            if chunk:
                content += chunk
                clear_output(wait=True)
                display(Markdown(content))

    print(f"\n\nNumber of tokens: {num_tokens}")
    if model_id:
        print(f"Model ID: {model_id}")
    else:
        print("Model ID not available in response")
