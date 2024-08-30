import os
from requests import request

def invoke(url="http://localhost:80", path="/", method="GET",
           headers={"Content-Type": "application/json"}, body=None,
           timeout=60):
    """
    Make an HTTP request to the specified URL.

    Args:
        url (str): Base URL for the request.
        path (str): Path to append to the URL.
        method (str): HTTP method to use.
        headers (dict): HTTP headers to include in the request.
        body (str, optional): Request body.
        timeout (int): Request timeout in seconds.

    Returns:
        requests.Response: The response from the server.
    """
    api_key = os.environ.get("API_KEY", "test_api_key")
    headers["X-API-Key"] = api_key
    return request(method, f"{url}{path}", headers=headers, data=body, timeout=timeout)

def test_list_endpoints():
    """
    Test the /list_endpoints API endpoint.
    """
    response = invoke(path="/list_endpoints", method="GET")
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

