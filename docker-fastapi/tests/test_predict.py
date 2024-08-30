import json
import os
import pprint
import pytest
from requests import request

@pytest.fixture
def api_key():
    """
    Fixture to get the API_KEY environment variable.

    Returns:
        str: The API key value.
    """
    print(os.environ.get("API_KEY"))
    return os.environ.get("API_KEY")

@pytest.fixture
def invalid_api_key():
    """
    Fixture to set an invalid API_KEY

    Returns:
        str: The API key value.
    """
    return "this_isnt_a_valid_key"

def invoke(url="http://localhost:80", path="/", method="GET",
           headers=None, body=None, timeout=60, api_key=None):
    """
    Make an HTTP request to the specified URL.

    Args:
        url (str): Base URL for the request.
        path (str): Path to append to the URL.
        method (str): HTTP method to use.
        headers (dict, optional): HTTP headers to include in the request.
        body (str, optional): Request body.
        timeout (int): Request timeout in seconds.
        api_key (str, optional): API key to use for the request.

    Returns:
        requests.Response: The response from the server.
    """
    if headers is None:
        headers = {"Content-Type": "application/json"}
    if api_key:
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
    response = invoke(path="/predict", method="POST", body=json.dumps(body), api_key=api_key)
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
    response = invoke(path="/predict", method="POST", body=json.dumps(missing_model_body), api_key=api_key)
    assert response.status_code == 200
 
def test_predict_missing_user(missing_user_body, api_key):
    response = invoke(path="/predict", method="POST", body=json.dumps(missing_user_body), api_key=api_key)
    assert response.status_code == 200

def test_predict_missing_system(missing_system_body, api_key):
    response = invoke(path="/predict", method="POST", body=json.dumps(missing_system_body), api_key=api_key)
    assert response.status_code == 200

#
# Error cases
#

def test_predict_none_body(api_key):
    response = invoke(path="/predict", method="POST", body=None, api_key=api_key)
    assert response.status_code == 500
    assert response.json()["detail"].startswith("Invalid JSON input:")

def test_predict_empty_body(api_key):
    response = invoke(path="/predict", method="POST", body=json.dumps({}), api_key=api_key)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["generated_text"] == ""
    assert response_json["details"]["finish_reason"] == "error"

def test_predict_invalid_json(invalid_json_body, api_key):
    response = invoke(path="/predict", method="POST", body=invalid_json_body, api_key=api_key)
    assert response.status_code == 500
    assert response.json()["detail"].startswith("Invalid JSON input:")

def test_predict_missing_messages(missing_messages_body, api_key):
    response = invoke(path="/predict", method="POST", body=json.dumps(missing_messages_body), api_key=api_key)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["generated_text"] == ""
    assert response_json["details"]["finish_reason"] == "error"

def test_predict_with_invalid_api_key(body, invalid_api_key):
    response = invoke(path="/predict", method="POST", body=json.dumps(body), api_key=invalid_api_key)
    assert response.status_code == 403
    assert response.json()["detail"].startswith("Could not validate credentials")

