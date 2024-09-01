"""Module for interacting with an OpenAI-compatible API endpoint."""

import os
import sys
import requests
import openai
import httpx
from openai import OpenAI
from invoke import invoke

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python invoke_endpoint.py <hostname> <port_number>")
        exit(1)

    API_KEY = os.environ.get("API_KEY")
    if API_KEY is None:
        print("API_KEY environment variable is not set.")
        exit(1)

    HOSTNAME = sys.argv[1]
    PORT_NUMBER = int(sys.argv[2])
    BASE_URL = f"https://{HOSTNAME}:{PORT_NUMBER}"
    print(f"Connecting to {BASE_URL}")

    response = invoke(url=BASE_URL, path="/list_endpoints")
    assert response.status_code == 200
    endpoints_data = response.json()
    assert len(endpoints_data) > 0, "No endpoints are currently in service"
    endpoint_name = endpoints_data[0]['EndpointName']
    print(f"Using endpoint: {endpoint_name}")

    # Create a custom HTTPX client to disable SSL verification
    client = httpx.Client(verify=False)

    # Set the custom client in the OpenAI library
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        http_client=client
    )

    response = client.chat.completions.create(
        model=endpoint_name,
        messages=[
            {"role": "system",
             "content": ("You are a helpful technical assistant giving detailed "
                         "and factual answers.")},
            {"role": "user",
             "content": "Why are transformers better models than LSTM?"}
        ],
        stream=False,
        max_tokens=500,
    )

    print(response.choices[0].message.content)
