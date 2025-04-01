"""Module for invoking API endpoints with various HTTP methods."""

import json
import os
import pprint
import sys

from invoke import invoke

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python invoke_endpoint.py <hostname> <port_number>")
        sys.exit(1)

    if os.environ.get("API_KEY") is None:
        print("API_KEY environment variable is not set.")
        sys.exit(1)

    HOSTNAME = sys.argv[1]
    PORT_NUMBER = int(sys.argv[2])

    URL = f"https://{HOSTNAME}:{PORT_NUMBER}"
    print(f"Connecting to {URL}")

    response = invoke(url=URL)
    assert response.status_code == 200
    pprint.pprint(response.json())

    response = invoke(url=URL, path="/list_endpoints")
    assert response.status_code == 200
    endpoints_data = response.json()
    pprint.pprint(response.json())
    assert len(endpoints_data) > 0, "No endpoints are currently in service"

    endpoint_name = endpoints_data[0]["EndpointName"]
    print(f"Using endpoint: {endpoint_name}")

    sample_request = {
        "model": endpoint_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "As a friendly technical assistant engineer, "
                    "answer the question in detail."
                ),
            },
            {
                "role": "user",
                "content": "Why are transformers better models than LSTM?",
            },
        ],
        "max_tokens": 256,
    }
    response = invoke(
        url=URL, path=f"/predict", method="POST", body=json.dumps(sample_request)
    )
    print(response.content)
    assert response.status_code == 200
    pprint.pprint(response.json())
