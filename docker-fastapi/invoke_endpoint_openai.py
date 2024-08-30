import os
from openai import OpenAI

API_KEY = os.environ.get("API_KEY")

headers = {}
headers["X-API-Key"] = API_KEY

client = OpenAI(
    base_url=f"http://localhost:80",
    api_key="blah",
)

chat_completion = client.chat.completions.create(
    model="scribe",
    messages=[
        {"role": "system", "content": "You are a helpful technical assistant giving detailed and factual answers." },
        {"role": "user", "content": "Why are transformers better models than LSTM?"}
    ],
    stream=False,
    max_tokens=500,
    extra_headers=headers
)

print(chat_completion.choices[0].message.content)
