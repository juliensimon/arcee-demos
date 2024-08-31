# pylint: disable=W0621


"""
This module contains test cases for the predict functionality.

It includes various fixtures and test functions to verify the behavior
of the prediction API under different scenarios.
"""

import json
import os

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


def invoke(
    url="http://localhost:80",
    path="/",
    method="GET",
    headers=None,
    body=None,
    timeout=60,
    api_key=None,
):
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
    return request(method, f"{url}{path}", headers=headers,
                   data=body, timeout=timeout)


@pytest.fixture
def body():
    """
    Fixture that provides a valid request body for the predict endpoint.

    Returns:
        dict: A dictionary containing model, messages, and max_tokens.
    """
    return {
        "model": "arcee-ai/Arcee-Scribe",
        "messages": [
            {
                "role": "system",
                "content": ("As a friendly technical assistant engineer, "
                            "answer the question in detail."),
            },
            {
                "role": "user",
                "content": "Why are transformers better models than LSTM?",
            },
        ],
        "max_tokens": 256,
    }


@pytest.fixture
def missing_model_body():
    """
    Fixture that provides a request body without a model specified.

    Returns:
        dict: A dictionary containing messages and max_tokens, but no model.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": ("As a friendly technical assistant engineer, "
                            "answer the question in detail."),
            },
            {
                "role": "user",
                "content": "Why are transformers better models than LSTM?",
            },
        ],
        "max_tokens": 256,
    }


@pytest.fixture
def missing_messages_body():
    """
    Fixture that provides a request body without messages.

    Returns:
        dict: A dictionary containing model and max_tokens, but no messages.
    """
    return {
        "model": "arcee-ai/Arcee-Scribe",
        "max_tokens": 256,
    }


@pytest.fixture
def missing_system_body():
    """
    Fixture that provides a request body without a system message.

    Returns:
        dict: A dictionary containing model, user message, and max_tokens,
        but no system message.
    """
    return {
        "model": "arcee-ai/Arcee-Scribe",
        "messages": [
            {
                "role": "user",
                "content": "Why are transformers better models than LSTM?",
            },
        ],
        "max_tokens": 256,
    }


@pytest.fixture
def missing_user_body():
    """
    Fixture that provides a request body without a user message.

    Returns:
        dict: A dictionary containing model, system message, and max_tokens,
        but no user message.
    """
    return {
        "model": "arcee-ai/Arcee-Scribe",
        "messages": [
            {
                "role": "system",
                "content": ("As a friendly technical assistant engineer, "
                            "answer the question in detail."),
            },
        ],
        "max_tokens": 256,
    }


@pytest.fixture
def invalid_json_body():
    """
    Fixture that provides an invalid JSON string as a request body.

    Returns:
        str: An invalid JSON string missing a comma after the "model" key.
    """
    return """{
        "model": "arcee-ai/Arcee-Scribe"
        "messages": [
            {"role": "system", "content": ("As a friendly technical assistant "
                                          "engineer, answer the question in "
                                          "detail.")},
            {"role": "user", "content": "Why are transformers better models "
                                          "than LSTM?"}
        ],
        "max_tokens": 256,
    }"""


#
# Success cases
#


def test_predict(body, api_key):
    """
    Test the predict endpoint with valid input.

    Args:
        body (dict): The request body containing model and message data.
        api_key (str): The API key for authentication.

    Asserts:
        - Response status code is 200
        - Response JSON contains expected keys and non-empty values
        - Completion tokens are greater than 0
    """
    response = invoke(
        path="/predict", method="POST", body=json.dumps(body), api_key=api_key
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json is not None
    assert "choices" in response_json
    assert len(response_json["choices"]) > 0
    assert "message" in response_json["choices"][0]
    assert "content" in response_json["choices"][0]["message"]
    assert response_json["choices"][0]["message"]["content"] is not None
    assert response_json["choices"][0]["message"]["content"] != ""
    assert response_json["usage"]["completion_tokens"] > 0


def test_predict_missing_model(missing_model_body, api_key):
    """
    Test the predict endpoint with a missing model in the request body.

    Args:
        missing_model_body (dict): The request body without a model specified.
        api_key (str): The API key for authentication.

    Asserts:
        - Response status code is 200
    """
    response = invoke(
        path="/predict",
        method="POST",
        body=json.dumps(missing_model_body),
        api_key=api_key,
    )
    assert response.status_code == 200


def test_predict_missing_user(missing_user_body, api_key):
    """
    Test the predict endpoint with a missing user message in the request body.

    Args:
        missing_user_body (dict): The request body without a user message.
        api_key (str): The API key for authentication.

    Asserts:
        - Response status code is 200
    """
    response = invoke(
        path="/predict",
        method="POST",
        body=json.dumps(missing_user_body),
        api_key=api_key,
    )
    assert response.status_code == 200


def test_predict_missing_system(missing_system_body, api_key):
    """
    Test the predict endpoint with a missing system message
    in the request body.

    Args:
        missing_system_body (dict): The request body without a system message.
        api_key (str): The API key for authentication.

    Asserts:
        - Response status code is 200
    """
    response = invoke(
        path="/predict",
        method="POST",
        body=json.dumps(missing_system_body),
        api_key=api_key,
    )
    assert response.status_code == 200


#
# Error cases
#


def test_predict_none_body(api_key):
    """
    Test the predict endpoint with a None body.

    Args:
        api_key (str): The API key for authentication.

    Asserts:
        - Response status code is 500
        - Response JSON contains an error message about invalid JSON input
    """
    response = invoke(path="/predict", method="POST",
                      body=None, api_key=api_key)
    assert response.status_code == 500
    assert response.json()["detail"].startswith("Invalid JSON input:")


def test_predict_empty_body(api_key):
    """
    Test the predict endpoint with an empty body.

    Args:
        api_key (str): The API key for authentication.

    Asserts:
        - Response status code is 200
        - Response JSON contains an empty generated text
        and an error finish reason
    """
    response = invoke(
        path="/predict", method="POST", body=json.dumps({}), api_key=api_key
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["generated_text"] == ""
    assert response_json["details"]["finish_reason"] == "error"


def test_predict_invalid_json(invalid_json_body, api_key):
    """
    Test the predict endpoint with an invalid JSON body.

    Args:
        invalid_json_body (str): An invalid JSON string.
        api_key (str): The API key for authentication.

    Asserts:
        - Response status code is 500
        - Response JSON contains an error message about invalid JSON input
    """
    response = invoke(
        path="/predict", method="POST", body=invalid_json_body, api_key=api_key
    )
    assert response.status_code == 500
    assert response.json()["detail"].startswith("Invalid JSON input:")


def test_predict_missing_messages(missing_messages_body, api_key):
    """
    Test the predict endpoint with missing messages in the request body.

    Args:
        missing_messages_body (dict): The request body without messages.
        api_key (str): The API key for authentication.

    Asserts:
        - Response status code is 200
        - Response JSON contains an empty generated text
        and an error finish reason
    """
    response = invoke(
        path="/predict",
        method="POST",
        body=json.dumps(missing_messages_body),
        api_key=api_key,
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["generated_text"] == ""
    assert response_json["details"]["finish_reason"] == "error"


def test_predict_with_invalid_api_key(body, invalid_api_key):
    """
    Test the predict endpoint with an invalid API key.

    Args:
        body (dict): The request body containing model and message data.
        invalid_api_key (str): An invalid API key.

    Asserts:
        - Response status code is 403
        - Response JSON contains an error message about invalid credentials
    """
    response = invoke(
        path="/predict",
        method="POST",
        body=json.dumps(body),
        api_key=invalid_api_key,
    )
    assert response.status_code == 403
    assert response.json()["detail"].startswith(
        "Could not validate credentials")
