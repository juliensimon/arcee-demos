"""Module for invoking API endpoints with various HTTP methods."""

import json
import os
import pprint

from invoke import invoke

HOSTNAME = "localhost"
PORT_NUMBER = 8000

if __name__ == "__main__":

    URL = f"https://{HOSTNAME}:{PORT_NUMBER}"
    print(URL)

    response = invoke(url=URL)
    assert response.status_code == 200
    pprint.pprint(response.json())

    response = invoke(url=URL, path="/list_endpoints")
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
