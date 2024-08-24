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
