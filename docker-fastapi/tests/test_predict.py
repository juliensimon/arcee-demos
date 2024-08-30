import json
import os
import pprint
import pytest
from requests import request


def invoke(url="http://localhost:80", path="/", method="GET",
           headers={"Content-Type": "application/json"}, body=None,
           timeout=60):
    """
    Make an HTTP request to the specified URL with the given parameters.

    Args:
        url (str): The base URL for the request.
        path (str): The path to append to the URL.
        method (str): The HTTP method to use.
        headers (dict): The headers to include in the request.
        body (str): The request body.
        timeout (int): The request timeout in seconds.

    Returns:
        Response: The response object from the request.
    """
    api_key = os.environ.get("API_KEY", "test_api_key")
    headers["X-API-Key"] = api_key
    return request(method, f"{url}{path}", headers=headers, data=body, timeout=timeout)


@pytest.fixture
def body():
    return {
        "model": "arcee-ai/Arcee-Scribe",
        "messages": [
            {"role": "system", "content": "As a friendly technical assistant engineer, answer the question in detail."},
            {"role": "user", "content": "Why are transformers better models than LSTM?"}
        ],
        "max_tokens": 256,
    }

@pytest.fixture
def missing_model_body():
    return {
        "messages": [
            {"role": "system", "content": "As a friendly technical assistant engineer, answer the question in detail."},
            {"role": "user", "content": "Why are transformers better models than LSTM?"}
        ],
        "max_tokens": 256,
    }

@pytest.fixture
def missing_messages_body():
    return {
        "model": "arcee-ai/Arcee-Scribe",
        "max_tokens": 256,
    }

@pytest.fixture
def missing_system_body():
    return {
        "model": "arcee-ai/Arcee-Scribe",
        "messages": [
            {"role": "user", "content": "Why are transformers better models than LSTM?"}
        ],
        "max_tokens": 256,
    }

@pytest.fixture
def missing_user_body():
    return {
        "model": "arcee-ai/Arcee-Scribe",
        "messages": [
            {"role": "system", "content": "As a friendly technical assistant engineer, answer the question in detail."},
        ],
        "max_tokens": 256,
    }

@pytest.fixture
def api_key():
    """
    Fixture to set and unset the API_KEY environment variable.

    Yields:
        str: The API key value.
    """
    os.environ["API_KEY"] = "test_api_key"
    yield "test_api_key"
    del os.environ["API_KEY"]

@pytest.fixture
def invalid_json_body():
    return '''{
        "model": "arcee-ai/Arcee-Scribe"
        "messages": [
            {"role": "system", "content": "As a friendly technical assistant engineer, answer the question in detail."},
            {"role": "user", "content": "Why are transformers better models than LSTM?"}
        ],
        "max_tokens": 256,
    }'''    

#
# Success cases
#

def test_predict(body, api_key):
    response = invoke(path="/predict", method="POST", body=json.dumps(body))
    assert response.status_code == 200
    response_json = response.json()
    assert response_json is not None
    assert 'choices' in response_json
    assert len(response_json['choices']) > 0
    assert 'message' in response_json['choices'][0]
    assert 'content' in response_json['choices'][0]['message']
    assert response_json['choices'][0]['message']['content'] is not None
    assert response_json['choices'][0]['message']['content'] != ""
    assert response_json['usage']['completion_tokens'] > 0

def test_predict_missing_model(missing_model_body, api_key):
    response = invoke(path="/predict", method="POST", body=json.dumps(missing_model_body))
    assert response.status_code == 200
 
def test_predict_missing_user(missing_user_body):
    response = invoke(path="/predict", method="POST", body=json.dumps(missing_user_body))
    assert response.status_code == 200


def test_predict_missing_system(missing_system_body):
    response = invoke(path="/predict", method="POST", body=json.dumps(missing_system_body))
    assert response.status_code == 200

#
# Error cases
#

def test_predict_none_body():
    response = invoke(path="/predict", method="POST", body=None)
    assert response.status_code == 500
    assert response.json()["detail"].startswith("Invalid JSON input:")

def test_predict_empty_body():
    response = invoke(path="/predict", method="POST", body=json.dumps({}))
    assert response.status_code == 500
    assert response.json()['detail'] == "Prediction error: 500: Payload is empty"

def test_predict_invalid_json(invalid_json_body):
    response = invoke(path="/predict", method="POST", body=invalid_json_body)
    assert response.status_code == 500
    assert response.json()["detail"].startswith("Invalid JSON input:")

def test_predict_missing_messages(missing_messages_body):
    response = invoke(path="/predict", method="POST", body=json.dumps(missing_messages_body))
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["generated_text"] == ""
    assert response_json["details"]["finish_reason"] == "error"

def test_predict_with_invalid_api_key():
    response = invoke(path="/predict", method="POST", body=json.dumps(body))
    assert response.status_code == 403

