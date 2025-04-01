"""FastAPI application for interacting with SageMaker endpoints."""

import json
import os

import boto3
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

APP_VERSION = "0.0.1"

ENDPOINTS = []


def get_env_var(var_name):
    """
    Get environment variable and assert it is set.

    Args:
        var_name (str): Name of the environment variable.

    Returns:
        str: Value of the environment variable.

    Raises:
        AssertionError: If the environment variable is not set.
    """
    var = os.environ.get(var_name)
    assert var is not None, f"{var_name} environment variable is not set"
    print(f"{var_name}: {var}")
    return var


API_KEY = get_env_var("API_KEY")
REGION_NAME = get_env_var("REGION_NAME")

app = FastAPI()
sm = boto3.client("sagemaker", region_name=REGION_NAME)
sm_rt = boto3.client("sagemaker-runtime", region_name=REGION_NAME)

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Validate the API key.

    Args:
        api_key (str): The API key from the request header.

    Returns:
        str: The validated API key.

    Raises:
        HTTPException: If the API key is invalid or missing.
    """
    if api_key == f"Bearer {API_KEY}":
        return api_key.split(" ")[1]
    raise HTTPException(status_code=403, detail="Could not validate credentials")


@app.get("/", dependencies=[Depends(get_api_key)])
def ping():
    """Return ping message"""
    return {"sagemaker-proxy": APP_VERSION}


@app.get("/list_endpoints", dependencies=[Depends(get_api_key)])
def list_endpoints():
    """List all SageMaker endpoints"""
    global ENDPOINTS
    endpoints = sm.list_endpoints()
    ENDPOINTS = [endpoint["EndpointName"] for endpoint in endpoints["Endpoints"]]
    endpoint_details = []
    for endpoint in endpoints["Endpoints"]:
        endpoint_description = sm.describe_endpoint(
            EndpointName=endpoint["EndpointName"]
        )
        endpoint_config = sm.describe_endpoint_config(
            EndpointConfigName=endpoint_description["EndpointConfigName"]
        )
        production_variant = endpoint_config["ProductionVariants"][0]
        model_name = production_variant["ModelName"]
        model_details = sm.describe_model(ModelName=model_name)
        endpoint_details.append(
            {
                "EndpointName": endpoint["EndpointName"],
                "EndpointStatus": endpoint["EndpointStatus"],
                "InstanceType": production_variant["InstanceType"],
                "Container": model_details["PrimaryContainer"]["Image"],
                "ModelEnvironment": model_details["PrimaryContainer"].get(
                    "Environment", {}
                ),
            }
        )
    return endpoint_details


@app.post("/predict", dependencies=[Depends(get_api_key)])
async def predict(request: Request):
    """
    Invoke the SageMaker endpoint.

    Args:
        request (Request): The request object containing the payload.
        endpoint_name (str): The name of the SageMaker endpoint.

    Returns:
        dict: The prediction result.

    Raises:
        HTTPException: If there is an error with the prediction.
    """
    try:
        global ENDPOINTS
        payload = await request.json()
        endpoint_name = payload["model"]
        if endpoint_name not in ENDPOINTS:
            # Refresh the list of endpoints
            endpoints = sm.list_endpoints()
            ENDPOINTS = [
                endpoint["EndpointName"] for endpoint in endpoints["Endpoints"]
            ]
            if endpoint_name not in ENDPOINTS:
                raise HTTPException(
                    status_code=404, detail=f"Endpoint {endpoint_name} not found"
                )

        response = sm_rt.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode())
        return result
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON input: {str(json_err)}"
        ) from json_err
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        ) from e


@app.post("/chat/completions", dependencies=[Depends(get_api_key)])
async def chat_completions(request: Request):
    """
    Invoke the SageMaker endpoint.

    Args:
        request (Request): The request object containing the payload.
        endpoint_name (str): The name of the SageMaker endpoint.

    Returns:
        dict: The prediction result.

    Raises:
        HTTPException: If there is an error with the prediction.
    """
    try:
        global ENDPOINTS
        payload = await request.json()
        endpoint_name = payload["model"]
        if endpoint_name not in ENDPOINTS:
            # Refresh the list of endpoints
            endpoints = sm.list_endpoints()
            ENDPOINTS = [
                endpoint["EndpointName"] for endpoint in endpoints["Endpoints"]
            ]
            if endpoint_name not in ENDPOINTS:
                raise HTTPException(
                    status_code=404, detail=f"Endpoint {endpoint_name} not found"
                )

        response = sm_rt.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode())
        return result
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON input: {str(json_err)}"
        ) from json_err
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="ssl/key.pem",
        ssl_certfile="ssl/cert.pem",
    )
