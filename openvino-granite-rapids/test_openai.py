from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="afm_45b_ov_int8",
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "Explain the attention mechanism in transformer models."}
    ],
    temperature=0.9,
    max_tokens=1024,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

