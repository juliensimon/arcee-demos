import os
from requests import request


def invoke(url="https://localhost:8000", path="/", method="GET",
           headers=None, body=None, timeout=60):
    """
    Invoke an endpoint with the given parameters.

    Args:
        url (str): The base URL of the endpoint.
        path (str): The path of the specific endpoint.
        method (str): The HTTP method to use.
        headers (dict): The headers to include in the request.
        body (str): The body of the request.
        timeout (int): The timeout for the request in seconds.

    Returns:
        requests.Response: The response from the endpoint.
    """
    if headers is None:
        headers = {}

    # Add Content-Type if not present
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    # Add API key to headers
    api_key = os.environ.get("API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        print("Warning: API_KEY environment variable is not set")

    return request(method, f"{url}{path}", headers=headers,
                   data=body, timeout=timeout, verify=False)

