import os

from requests import request

API_KEY = os.environ.get("API_KEY")


def invoke(
    url="https://localhost:8000",
    path="/",
    method="GET",
    headers=None,
    body=None,
    timeout=60,
    api_key=API_KEY,
):
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

    headers["Content-Type"] = "application/json"
    headers["Authorization"] = f"Bearer {api_key}"

    return request(
        method,
        f"{url}{path}",
        headers=headers,
        data=body,
        timeout=timeout,
        verify=False,
    )
