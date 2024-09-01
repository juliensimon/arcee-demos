"""Module for interacting with an OpenAI-compatible API endpoint."""

import os

from openai import OpenAI

HOSTNAME = "ec2-35-87-44-203.us-west-2.compute.amazonaws.com"
PORT = 8000

API_KEY = os.environ.get("API_KEY")

headers = {}
headers["X-API-Key"] = API_KEY

BASE_URL = f"https://{HOSTNAME}:{PORT}"

client = OpenAI(
    base_url=BASE_URL,
    api_key="blah",
)

chat_completion = client.chat.completions.create(
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
    extra_headers=headers
)

print(chat_completion.choices[0].message.content)
