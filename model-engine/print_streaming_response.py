from IPython.display import Markdown, clear_output, display


def print_streaming_response(response):
    num_tokens = 0
    content = ""
    model_id = None

    if response is None:
        print("Response is None")
        return

    for message in response:
        if message is None:
            continue
            
        # Safely check for choices attribute
        if not hasattr(message, 'choices'):
            continue
            
        # Safely handle choices being None
        choices = getattr(message, 'choices', None)
        if choices is None or not choices:  # Empty list or None
            continue
            
        # Capture model ID from the first chunk if available
        if model_id is None and hasattr(message, "model"):
            model_id = message.model
            
        # Get the first choice safely
        choice = choices[0] if choices else None
        if choice is None:
            continue
            
        # Check for delta attribute
        if not hasattr(choice, 'delta'):
            continue
            
        delta = choice.delta
        if delta is None:
            continue
            
        # Check for content attribute
        if not hasattr(delta, 'content'):
            continue
            
        chunk = delta.content
        if chunk:
            num_tokens += 1
            content += chunk
            clear_output(wait=True)
            display(Markdown(content))

    print(f"\n\nNumber of tokens: {num_tokens}")
    if model_id:
        print(f"Model ID: {model_id}")
    else:
        print("Model ID not available in response")
