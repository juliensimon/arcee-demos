import boto3
import json
import os
from fastapi import FastAPI, Request

app = FastAPI()

endpoint_name = os.environ.get("ENDPOINT_NAME")
region_name = os.environ.get("REGION_NAME")

print(f"Endpoint name: {endpoint_name}")
print(f"Region name: {region_name}")

sm = boto3.client("sagemaker-runtime", region_name=region_name)


@app.get("/")
def ping():
    ''' Return ping message'''
    return {"endpoint name": endpoint_name, "region name": region_name}

@app.post("/update_endpoint")
async def update_endpoint(request: Request):
    global endpoint_name
    try:
        payload = await request.json()
        new_endpoint_name = payload.get("endpoint_name")
        if new_endpoint_name:
            endpoint_name = new_endpoint_name
            return {"message": f"Endpoint name updated to: {endpoint_name}"}
        else:
            return {"error": "No endpoint_name provided in the request body"}, 400
    except json.JSONDecodeError as json_err:
        return {"error": f"Invalid JSON input: {str(json_err)}"}, 400
    except Exception as e:
        return {"error": f"Error setting endpoint: {str(e)}"}, 500


@app.post("/predict")
async def predict(request: Request):
    ''' Invoke the SageMaker endpoint'''
    try:
        payload = await request.json()
        response = sm.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        result = json.loads(response['Body'].read().decode())
        return result
    except json.JSONDecodeError as json_err:
        return {"error": f"Invalid JSON input: {str(json_err)}"}, 400
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}, 500