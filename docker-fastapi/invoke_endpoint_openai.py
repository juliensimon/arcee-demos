"""Module for interacting with an OpenAI-compatible API endpoint."""

import os
import requests
import openai
import httpx
from openai import OpenAI

HOSTNAME = "localhost"
PORT = 8000

API_KEY = os.environ.get("API_KEY")

BASE_URL = f"https://{HOSTNAME}:{PORT}"

# Create a custom HTTPX client to disable SSL verification
client = httpx.Client(verify=False)

# Set the custom client in the OpenAI library
client = OpenAI(
        base_url = BASE_URL,
        api_key = API_KEY,
        http_client = client)

response = client.chat.completions.create(
    model="scribe",
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
