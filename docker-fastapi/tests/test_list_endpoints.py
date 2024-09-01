# pylint: disable=W0621

"""Tests for the list_endpoints API endpoint."""

import os

import pytest
from requests import request
from invoke import invoke

@pytest.fixture
def api_key():
    """
    Fixture to get the API_KEY environment variable.

    Returns:
        str: The API key value.
    """
    api_key = os.environ.get("API_KEY")
    if api_key is None:
        raise ValueError("API_KEY environment variable is not set")
    return api_key

def test_list_endpoints(api_key):
    """
    Test the /list_endpoints API endpoint.
    """
    response = invoke(path="/list_endpoints", method="GET",
                      api_key=api_key)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    for endpoint in data:
        assert 'EndpointName' in endpoint
        assert isinstance(endpoint['EndpointName'], str)

        assert 'InstanceType' in endpoint
        assert isinstance(endpoint['InstanceType'], str)

        assert 'Container' in endpoint
        assert isinstance(endpoint['Container'], str)

        assert 'ModelEnvironment' in endpoint
        assert isinstance(endpoint['ModelEnvironment'], dict)


def test_list_endpoints_with_invalid_api_key():
    """
    Test the /list_endpoints API endpoint with an invalid API key.
    """
    response = invoke(path="/list_endpoints", method="GET",
                      api_key="invalid_api_key")
    assert response.status_code == 403
    print(response.json())
