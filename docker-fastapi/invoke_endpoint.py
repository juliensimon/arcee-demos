import json
import pprint
from requests import request


def invoke(url="http://localhost:80", path="/", method="GET",
           headers={"Content-Type": "application/json"}, body=None,
           timeout=60):
    return request(method, f"{url}{path}", headers=headers, data=body, timeout=timeout)


if __name__ == "__main__":
    response = invoke()
    assert response.status_code == 200
    pprint.pprint(response.json())

    response = invoke(path="/list_endpoints", method="GET")
    assert response.status_code == 200
    pprint.pprint(response.json())

  

    body = {
        "model": "arcee-ai/Arcee-Scribe",
        "messages": [
            {"role": "system", "content": "As a friendly technical assistant engineer, answer the question in detail."},
            {"role": "user", "content": "Why are transformers better models than LSTM?"}
        ],
        "max_tokens": 256,
    }
    response = invoke(path="/predict", method="POST", body=json.dumps(body))
    assert response.status_code == 200
    pprint.pprint(response.json())

