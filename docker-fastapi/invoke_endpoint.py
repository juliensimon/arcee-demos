"""Module for invoking API endpoints with various HTTP methods."""

# flake8: noqa: E501

import os
import json
import pprint
from requests import request


def invoke(url="http://localhost:80", path="/", method="GET",
           headers=None, body=None, timeout=60):
    """
    Invoke an endpoint with the given parameters.

    Args:
        url (str): The base URL of the endpoint.
        path (str): The path of the specific endpoint.
        method (str): The HTTP method to use.
        headers (dict): The headers to include in the request.
        body (str): The body of the request.
        timeout (int): The timeout for the request in seconds.

    Returns:
        requests.Response: The response from the endpoint.
    """
    if headers is None:
        headers = {}

    # Add Content-Type if not present
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    # Add API key to headers
    api_key = os.environ.get("API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    else:
        print("Warning: API_KEY environment variable is not set")

    return request(method, f"{url}{path}", headers=headers, data=body, timeout=timeout)


if __name__ == "__main__":

    URL = "http://ec2-35-93-22-81.us-west-2.compute.amazonaws.com:8000"

    response = invoke(url=URL)
    assert response.status_code == 200
    pprint.pprint(response.json())

    response = invoke(url=URL, path="/list_endpoints", method="GET")
    assert response.status_code == 200
    pprint.pprint(response.json())

    sample_request = {
        "model": "arcee-ai/Arcee-Scribe",
        "messages": [
            {
                "role": "system",
                "content": ("As a friendly technical assistant engineer, "
                            "answer the question in detail.")
            },
            {
                "role": "user",
                "content": "Why are transformers better models than LSTM?"
            }
        ],
        "max_tokens": 256,
    }
    response = invoke(url=URL, path="/predict", method="POST",
                      body=json.dumps(sample_request))
    assert response.status_code == 200
    pprint.pprint(response.json())
