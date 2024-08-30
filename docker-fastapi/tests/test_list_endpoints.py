import os
import pytest
from requests import request

@pytest.fixture
def api_key():
    """
    Fixture to provide the API_KEY environment variable.

    Returns:
        str: The API key value.
    """
    return os.environ.get("API_KEY")

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

def test_list_endpoints(api_key):
    """
    Test the /list_endpoints API endpoint.
    """
    response = invoke(path="/list_endpoints", method="GET", api_key=api_key)
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
    response = invoke(path="/list_endpoints", method="GET", api_key="invalid_api_key")
    assert response.status_code == 403


